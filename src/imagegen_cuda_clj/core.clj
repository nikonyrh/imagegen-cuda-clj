(ns imagegen-cuda-clj.core
  (:require [uncomplicate.commons.core :as commons.c]
            [uncomplicate.clojurecuda.core
               :as clojurecuda.c
               :refer [memcpy-host! mem-alloc]]
            
            [clojure.repl :refer [source doc]]
            
            [mikera.image.core :as img]))


(defn make-fn! [source] ; Must be executed in a CUDA context!
  (let [name (->> source (re-find #"__global__[ ]+void[ ]*([^ ]+)[ \r\n\t]*\(") second)]
    (-> source
        clojurecuda.c/program
        clojurecuda.c/compile!
        clojurecuda.c/module
        (clojurecuda.c/function name))))


(defn launch! [kernel-fn grid & parameters] ; Must be executed in a CUDA context!
  (clojurecuda.c/launch! kernel-fn grid (apply clojurecuda.c/parameters parameters)))


(comment
  (clojurecuda.c/init)
  (->> (clojurecuda.c/device-count) range (map clojurecuda.c/device) (map commons.c/info) clojure.pprint/pprint)
  
  (def gpu (clojurecuda.c/device 0))
  
  
  (def sample-source
    "extern \"C\" __global__ void grayscale_img (const int res, unsigned char *im) {
       const int x = blockIdx.x * blockDim.x + threadIdx.x;
       const int y = blockIdx.y * blockDim.y + threadIdx.y;
       
       if (x < res && y < res) {
         im[y * res + x] = (x + 2 * y) & 0xFF;
       }
     };")
  
  (def res 256)
  (def image (java.awt.image.BufferedImage. res res java.awt.image.BufferedImage/TYPE_BYTE_GRAY))
  
  (clojurecuda.c/with-context (clojurecuda.c/context gpu)
    (let [buffer-cpu   (-> image .getRaster .getDataBuffer .getData)
          n-elems      (count buffer-cpu)
          output-gpu   (mem-alloc n-elems)
          
          kernel-fn    (make-fn! sample-source)
          _            (launch! kernel-fn (clojurecuda.c/grid-2d res res) res output-gpu)
          result-cpu   (memcpy-host! output-gpu (byte-array n-elems))]
      (System/arraycopy result-cpu 0 buffer-cpu 0 n-elems)))
  
  
  (img/show image :title "Image"))


(comment
  (take 10 (-> image .getRaster .getDataBuffer .getData))
  (-> image .getRaster .getDataBuffer .getData count))
  

(defn -main []
  (println "Hello, World!"))
