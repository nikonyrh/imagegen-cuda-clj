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
(def ^java.util.concurrent.ArrayBlockingQueue fn-queue     (java.util.concurrent.ArrayBlockingQueue. 1))
(def ^java.util.concurrent.ArrayBlockingQueue result-queue (java.util.concurrent.ArrayBlockingQueue. 1))

(def ^java.lang.Thread fn-queue-prosessor
  (let [thread
        (Thread.
          #(let [ctx (context gpu)]
             (with-context ctx
               (loop []
                 (when-let [^Callable f (.poll fn-queue 60000 java.util.concurrent.TimeUnit/MILLISECONDS)]
                   (assert (zero? (.size result-queue)))
                   (.add result-queue {:result (f)}))
                 (recur))))
          "fn-queue-prosessor")]
    (.start thread)
    thread))


(defn push-cuda-fn [^Callable f]
  (loop []
    (if (.offer fn-queue f)
      (let [result (.poll result-queue 1000 java.util.concurrent.TimeUnit/MILLISECONDS)]
        (assert result)
        (:result result))
      (do
        (assert (not= java.lang.Thread$State/TERMINATED (.getState fn-queue-prosessor)))
        (println "waiting for push-cuda-fn")
        (Thread/sleep 100)))))



(comment
  (push-cuda-fn #(assert false))
  
  [(.getId (Thread/currentThread))
   (push-cuda-fn #(.getId (Thread/currentThread)))]

  (push-cuda-fn #(+ 1 2 3))
  (time (push-cuda-fn #(do (Thread/sleep 500) (println "jee")))))



(defmacro cudafn [name args & body]
  `(defn ~name ~args
    (let [f# (fn [] ~@body)]
      (push-cuda-fn f#))))


; TODO: A macro for this?
(cudafn memcpy-host! [& args] (apply cc.core/memcpy-host! args))
(cudafn mem-alloc    [& args] (apply cc.core/mem-alloc    args))


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
  
  
  (def sample-source
    "extern \"C\" __global__ void grayscale_img (const int res, unsigned char *im) {
       const int x = blockIdx.x * blockDim.x + threadIdx.x;
       const int y = blockIdx.y * blockDim.y + threadIdx.y;
       
       if (x < res && y < res) {
         im[y * res + x] = (x + y + y * y / 32) & 0xFF;
       }
     };")
  
  
  (def res 256)
  (def ^java.awt.image.BufferedImage image (java.awt.image.BufferedImage. res res java.awt.image.BufferedImage/TYPE_BYTE_GRAY))
  
  (let [^java.awt.image.DataBufferByte buffer (-> image .getRaster .getDataBuffer)
        data-cpu     (-> buffer .getData)
        n-elems      (count data-cpu)
        output-gpu   (mem-alloc n-elems)
        
        kernel-fn    (make-fn! sample-source)
        _            (launch! kernel-fn (cc.core/grid-2d res res) res output-gpu)
        result-cpu   (memcpy-host! output-gpu (byte-array n-elems))]
    (System/arraycopy result-cpu 0 data-cpu 0 n-elems))
  
  
  (img/show image :title "Image"))


(comment
  (take 10 (-> image .getRaster .getDataBuffer .getData))
  (-> image .getRaster .getDataBuffer .getData count))
  

(defn -main []
  (println "Hello, World!"))
