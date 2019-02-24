(ns imagegen-cuda-clj.core
  (:require [uncomplicate.commons.core :as commons.core]
            [uncomplicate.clojurecuda.core :as cc.core :refer [with-context context]]            
            [clojure.repl :refer [source doc]]
            [clojure.pprint :refer [pprint]]
            
            [mikera.image.core :as img]))


(set! *warn-on-reflection* true)


(cc.core/init)
(def gpu (cc.core/device 0))

; I am not sure if we are heading to a better place, or worse...
(def ^java.util.concurrent.ArrayBlockingQueue fn-queue (java.util.concurrent.ArrayBlockingQueue. 1))

(def ^java.lang.Thread fn-queue-prosessor
  (let [thread
        (-> #(with-context (context gpu)
               (loop []
                 (when-let [[^Callable f result-p] (.poll fn-queue 60000 java.util.concurrent.TimeUnit/MILLISECONDS)]
                   (try (deliver result-p (f))
                     (catch Exception e
                       (deliver result-p e)
                       (throw e))))
                 (recur)))
          (Thread. "fn-queue-prosessor"))]
    (.start thread)
    thread))


(defn call-cuda-fn [^Callable f]
  (loop []
    (assert (not= java.lang.Thread$State/TERMINATED (.getState fn-queue-prosessor)))
    (let [result-p (promise)]
      (if (.offer fn-queue [f result-p])
        (let [result (deref result-p 1000 :timeout)]
          (assert (not= result :timeout))
          (when (instance? java.lang.Exception result)
            (println "Exception")
            (throw result))
          result)
        (do
          (println "retrying offer for call-cuda-fn")
          (Thread/sleep 100)
          (recur))))))



(comment
  (call-cuda-fn #(/ 0))
  
  [(.getId (Thread/currentThread))
   (call-cuda-fn #(.getId (Thread/currentThread)))]

  (call-cuda-fn #(+ 1 2 3))
  (time (call-cuda-fn #(do (Thread/sleep 500) (println "jee")))))



(defmacro cudafn [name args & body]
  `(defn ~name ~args
    (call-cuda-fn (fn [] ~@body))))

(defmacro cudafn-wrap [name]
  `(cudafn ~name  [& ~'args] (apply ~(symbol (str "cc.core/" name)) ~'args)))


(cudafn-wrap memcpy-host!)
(cudafn-wrap mem-alloc)


(cudafn make-fn! [source]
  (let [name (->> source (re-find #"__global__[ ]+void[ ]*([^ ]+)[ \r\n\t]*\(") second)]
    (-> source
        cc.core/program
        cc.core/compile!
        cc.core/module
        (cc.core/function name))))

(cudafn launch! [kernel-fn grid & parameters]
  (cc.core/launch! kernel-fn grid (apply cc.core/parameters parameters)))


(comment
  (->> (cc.core/device-count) range (map cc.core/device) (map commons.core/info) clojure.pprint/pprint)
  
  (def res 1024)
  (def ^java.awt.image.BufferedImage image (java.awt.image.BufferedImage. res res java.awt.image.BufferedImage/TYPE_BYTE_GRAY))
  
  (let [^java.awt.image.DataBufferByte buffer (-> image .getRaster .getDataBuffer)]
    (->> buffer .getBankData (map count)))
  
  (def sample-source "
    extern \"C\" __device__ unsigned int rgb (const unsigned int r, const unsigned int g, const unsigned int b) {
       return 0xFF; // (0xFF << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
    }
    
    extern \"C\" __global__ void rgb_img (const int res, unsigned char *im) {
       const int x = blockIdx.x * blockDim.x + threadIdx.x;
       const int y = blockIdx.y * blockDim.y + threadIdx.y;
       
       if (x < res && y < res) {
         //im[y * res + x] = rgb(x + y * y / 256, 2 * x + y, x * x / 256 + y);
         im[y * res + x] = (x + 2 * y) & 0xFF; // rgb(x, y, x + y);
       }
     };")
  
  
  (let [^java.awt.image.DataBufferByte buffer (-> image .getRaster .getDataBuffer)
        data-cpu     (-> buffer .getData)
        n-elems      (count data-cpu)
        output-gpu   (mem-alloc n-elems)
        
        kernel-fn    (make-fn! sample-source)
        _            (launch! kernel-fn (cc.core/grid-2d res res 32 32) res output-gpu)
        result-cpu   (memcpy-host! output-gpu (byte-array n-elems))]
    (System/arraycopy result-cpu 0 data-cpu 0 n-elems))
  
  (println (System/nanoTime))
  (img/show image :title "Image"))


(comment
  (take 10 (-> image .getRaster .getDataBuffer .getData))
  (-> image .getRaster .getDataBuffer .getData count))
  

(defn -main []
  (println "Hello, World!"))
