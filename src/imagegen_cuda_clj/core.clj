(ns imagegen-cuda-clj.core
  (:require [uncomplicate.commons.core :as commons.core]
            [uncomplicate.clojurecuda.core :as cc.core :refer [with-context context]]            
            [clojure.repl :refer [source doc]]
            [clojure.pprint :refer [pprint]]
            
            [mikera.image.core :as img]
            
            [nikonyrh-utilities-clj.core :as u]))


(set! *warn-on-reflection* true)


(cc.core/init)
(def gpu (cc.core/device 0))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; I am not sure if we are heading to a better place, or worse...
(def ^java.util.concurrent.ArrayBlockingQueue fn-queue (java.util.concurrent.ArrayBlockingQueue. 32))

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
  (assert (not= java.lang.Thread$State/TERMINATED (.getState fn-queue-prosessor)))
  (let [result-p (promise)
        _ (assert (.offer fn-queue [f result-p]) "fn-queue is full!")
        result (deref result-p 1000 :timeout)]
    (assert (not= result :timeout))
    (if (instance? java.lang.Exception result)
      (throw result)
      result)))



(comment
  ; To stop the fn-queue-prosessor thread...
  (call-cuda-fn #(/ 0))
  
  ; To confirm that we are running our CUDA calls consistently in the same thread,
  ; ref. https://dragan.rocks/articles/18/Interactive-GPU-Programming-3-CUDA-Context
  (let [thread-id-fn #(.getId (Thread/currentThread))]
    [(thread-id-fn) (call-cuda-fn thread-id-fn)])
 
  ; A few simple tests
  (call-cuda-fn #(+ 1 2 3))
  (time (call-cuda-fn #(do (Thread/sleep 500) (println "yes!")))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defmacro cudafn [name args & body]
  `(defn ~name ~args
    (call-cuda-fn (fn [] ~@body))))

(defmacro cudafn-wrap [name]
  `(cudafn ~name  [& ~'args] (apply ~(symbol (str "cc.core/" name)) ~'args)))


(cudafn-wrap memcpy-host!)
(cudafn-wrap synchronize!)
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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def kernel-fn
  (make-fn!
    (let [eps "1e-6f"
          pi  "3.141592653589793f"
          pi2 "6.283185307179586f"]
      (u/my-format "
    extern \"C\" __device__ int rgb(const int r, const int g, const int b) {
      return (0xFF << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
    }
    
    extern \"C\" __device__ float to_log(const int i, const int res) {
      const float f = ((float) i) / ((float) (res - 1)) * (1.0f - 2 * {%s:eps}) + {%s:eps};
      return logf(f / (1.0f - f));
    }
    
    extern \"C\" __device__ float to_normal(const float l) {
      return 1.0f / (1.0f + expf(-l));
    }
    
    extern \"C\" __global__ void rgb_img(const int res, int *im) {
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      const int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      const int subres = res / 4, bx = x / subres, by = y / subres;
      
      if (x < res && y < res) {
        const float
          i = to_log(x & (subres - 1), subres),
          j = to_log(y & (subres - 1), subres),
          
          d = expf(-0.35f * (i * i + j * j)),
          
          k1 = 1.0f + 2.0f * bx,
          k2 = 0.5f + 0.5f * by,
          
          r = to_normal(i + k1 * d * sinf(j * k2 * {%s:pi2})),
          g = to_normal(j + k1 * d * cosf(i * k2 * {%s:pi2})),
          b = to_normal((i + j) * 0.5f);
        
        im[y * res + x] = rgb(r * 255.0f + 0.5f, g * 255.0f + 0.5f, b * 255.0f + 0.5f);
      }
    }
"))))



(defn ^java.awt.image.BufferedImage make-renderer [res]
  ; Initializing resources and returning a closure.
  (let [^java.awt.image.BufferedImage image (java.awt.image.BufferedImage. res res java.awt.image.BufferedImage/TYPE_INT_RGB)
        ^java.awt.image.DataBufferInt buffer (-> image .getRaster .getDataBuffer)
        bytes-per-elem 4
        data-cpu    (-> buffer .getData)
        n-elems     (count data-cpu)
        output-gpu  (mem-alloc (* bytes-per-elem n-elems))]
   (fn [] ; kernel arguments would go here
     (launch! kernel-fn (cc.core/grid-2d res res 16 16) res output-gpu)
     (memcpy-host! output-gpu data-cpu)
     image)))


(comment
  (->> (cc.core/device-count) range (map cc.core/device) (map commons.core/info) clojure.pprint/pprint)
  
  (def render-fn (make-renderer 1024))
  
  (def image (render-fn))
  (img/show image :title "Image"))


(comment
  (println "")
  (doseq [res (for [i (range 6 15)] (bit-shift-left 1 i))]
    (let [render-fn (make-renderer res)
          n-iter    20
          t-mean
          (->> (for [_ (range n-iter)]
                 (let [t0 (System/nanoTime)
                       _  (render-fn)
                       t1 (System/nanoTime)]
                   (- t1 t0)))
               (apply +)
               (* 1e-6 (/ n-iter)))]
      (println (u/my-format "Done in {%10.3f:t-mean} ms ({%4d:res} x {%4d:res})")))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn -main []
  (let [render-fn (make-renderer 1024)
        image     (render-fn)
        file      (java.io.File. "image.jpg")]
    (javax.imageio.ImageIO/write image "JPG" file)))
