(ns imagegen-cuda-clj.core
  (:require [uncomplicate.commons.core :as commons.c]
            [uncomplicate.clojurecuda.core :as clojurecuda.c]))


; At this point I have no clue if this is a good idea...
(defn make-kernel-source [name args source]
  (format "extern \"C\" __global__ void %s (%s) { %s };" name args source))


(defn make-fn
  ([source]
   (let [name (->> source (re-find #"__global__[ ]+void[ ]*([^ ]+)[ \r\n\t]*\(") second)]
     (make-fn name source)))
  ([name source]
   (-> source
       clojurecuda.c/program
       clojurecuda.c/compile!
       clojurecuda.c/module
       (clojurecuda.c/function name)))
  ([name args source]
   (->> (make-kernel-source name args source)
        (make-fn name))))


(comment
  (clojurecuda.c/init)
  (->> (clojurecuda.c/device-count) range (map clojurecuda.c/device) (map commons.c/info) clojure.pprint/pprint)
  
  (def sample-source
    (make-kernel-source
      "example_kernel", "const int n, const float *a, float *b"
      
      "int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) {
         b[i] = a[i] + 1.0f; // <-- math!
       }"))
  
  (def gpu (clojurecuda.c/device 0))
  
  (clojurecuda.c/with-context (clojurecuda.c/context gpu)
    (let [kernel-fn (make-fn sample-source)
          input-buffer (float-array (range n-elems)) ; 0, 1, 2, ...
          input-gpu    (clojurecuda.c/mem-alloc n-elems)
          output-gpu    (clojurecuda.c/mem-alloc n-elems)]
      
      (clojurecuda.c/memcpy-host! input-buffer input-gpu)
      (clojurecuda.c/launch! kernel-fn (clojurecuda.c/grid-1d n-elems) (clojurecuda.c/parameters n-elems input-gpu output-gpu))
      
      (def output-cpu (clojurecuda.c/memcpy-host! output-gpu (float-array n-elems)))))
  
  (take 10 output-cpu))
  

(defn -main []
  (println "Hello, World!"))
