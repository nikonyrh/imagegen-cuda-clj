(ns imagegen-cuda-clj.core
  (:require [uncomplicate.commons.core :as commons.c]
            [uncomplicate.clojurecuda.core :as clojurecuda.c]
            [clojure.repl :refer [source doc]]))


(defn make-fn [source] ; Must be executed in a CUDA context!
  (let [name (->> source (re-find #"__global__[ ]+void[ ]*([^ ]+)[ \r\n\t]*\(") second)]
    (-> source
        clojurecuda.c/program
        clojurecuda.c/compile!
        clojurecuda.c/module
        (clojurecuda.c/function name))))


(comment
  (doc clojurecuda.c/program)
  (doc clojurecuda.c/compile!)
  (doc clojurecuda.c/module)
  (doc clojurecuda.c/function)
  (doc clojurecuda.c/parameters)
  
  (clojurecuda.c/init)
  (->> (clojurecuda.c/device-count) range (map clojurecuda.c/device) (map commons.c/info) clojure.pprint/pprint)
  
  (def gpu (clojurecuda.c/device 0))
  
  
  (def sample-source
    "/* extern \"C\" __device__ float innercall (const float x, const int y) {
       return x + y;
     }; */
     
     extern \"C\" __global__ void example_kernel (const int n, const float *a, float *b, const float c) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) {
         b[i] = a[i] + c; // innercall(a[i], c)
       }
     };")
  
  
  (clojurecuda.c/with-context (clojurecuda.c/context gpu)
    (let [kernel-fn    (make-fn sample-source)
          input-buffer (float-array (range n-elems)) ; 0, 1, 2, ...
          input-gpu    (clojurecuda.c/mem-alloc n-elems)
          output-gpu   (clojurecuda.c/mem-alloc n-elems)]
      
      (clojurecuda.c/memcpy-host! input-buffer input-gpu)
      (clojurecuda.c/launch! kernel-fn (clojurecuda.c/grid-1d n-elems) (clojurecuda.c/parameters n-elems input-gpu output-gpu (Float. 1.23)))
      
      (def output-cpu (clojurecuda.c/memcpy-host! output-gpu (float-array n-elems)))))
  
  
  (take 10 output-cpu))
  

(defn -main []
  (println "Hello, World!"))
