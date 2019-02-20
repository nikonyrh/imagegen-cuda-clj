(ns imagegen-cuda-clj.core
  (:require [uncomplicate.commons.core :as commons.c]
            [uncomplicate.clojurecuda.core
               :as clojurecuda.c
               :refer [memcpy-host! mem-alloc]]
            [clojure.repl :refer [source doc]]))


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
    "extern \"C\" __device__ float innercall1 (const float x, const float y, const int i) {
       return x + y * i;
     };

     extern \"C\" __global__ void example_kernel (const int n, const float *a, float *b, const float c) {
       int i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) {
         b[i] = innercall1(a[i], c, i);
       }
     };")
  
  
  (clojurecuda.c/with-context (clojurecuda.c/context gpu)
    (let [kernel-fn    (make-fn! sample-source)
          input-cpu    (float-array (range n-elems)) ; 0, 1, 2, ...
          input-gpu    (mem-alloc n-elems)
          output-gpu   (mem-alloc n-elems)]
      
      (memcpy-host! input-cpu input-gpu)
      (launch! kernel-fn (clojurecuda.c/grid-1d n-elems) n-elems input-gpu output-gpu (Float. 1.23))
      
      (def output-cpu (memcpy-host! output-gpu (float-array n-elems)))))
  
  
  (take 10 output-cpu))
  

(defn -main []
  (println "Hello, World!"))
