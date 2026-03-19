#!/usr/bin/env bb
;; SPI Virtuoso — Babashka/JVM racer
;; Tricks: pmap, virtual threads, unchecked-math, type hints
;; Targets: same SplitMix64 XOR fingerprint as Zig/Julia/Swift racers
;;
;; bb clojure-racer.bb

(set! *unchecked-math* true)

(def ^:const GOLDEN (unchecked-long 0x9e3779b97f4a7c15))
(def ^:const MIX1   (unchecked-long 0xbf58476d1ce4e5b9))
(def ^:const MIX2   (unchecked-long 0x94d049bb133111eb))
(def ^:const SEED   (long 42))

(defn sm64 ^long [^long seed ^long index]
  (let [z (unchecked-add seed (unchecked-multiply GOLDEN index))
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 30)) MIX1)
        z (unchecked-multiply (bit-xor z (unsigned-bit-shift-right z 27)) MIX2)]
    (bit-xor z (unsigned-bit-shift-right z 31))))

(defn extract-rgb ^long [^long val]
  (bit-or
    (bit-shift-left (bit-and (unsigned-bit-shift-right val 16) 0xFF) 16)
    (bit-or
      (bit-shift-left (bit-and (unsigned-bit-shift-right val 8) 0xFF) 8)
      (bit-and val 0xFF))))

;; L0: Scalar loop
(defn l0-scalar ^long [^long n]
  (loop [i (long 0) xor (long 0)]
    (if (< i n)
      (recur (unchecked-inc i) (bit-xor xor (sm64 SEED i)))
      xor)))

;; L1: 8-wide unrolled (JVM can't vectorize but unroll hides latency)
(defn l1-pipeline8 ^long [^long n]
  (let [n8 (bit-and n (bit-not 7))]
    (loop [i (long 0)
           a0 (long 0) a1 (long 0) a2 (long 0) a3 (long 0)
           b0 (long 0) b1 (long 0) b2 (long 0) b3 (long 0)]
      (if (< i n8)
        (recur (unchecked-add i 8)
               (bit-xor a0 (sm64 SEED i))
               (bit-xor a1 (sm64 SEED (unchecked-add i 1)))
               (bit-xor a2 (sm64 SEED (unchecked-add i 2)))
               (bit-xor a3 (sm64 SEED (unchecked-add i 3)))
               (bit-xor b0 (sm64 SEED (unchecked-add i 4)))
               (bit-xor b1 (sm64 SEED (unchecked-add i 5)))
               (bit-xor b2 (sm64 SEED (unchecked-add i 6)))
               (bit-xor b3 (sm64 SEED (unchecked-add i 7))))
        (let [base (bit-xor a0 (bit-xor a1 (bit-xor a2 (bit-xor a3
                     (bit-xor b0 (bit-xor b1 (bit-xor b2 b3)))))))]
          (loop [j i result base]
            (if (< j n)
              (recur (unchecked-inc j) (bit-xor result (sm64 SEED j)))
              result)))))))

;; L2: Fused gen+RGB+XOR
(defn l2-fused ^long [^long n]
  (let [n8 (bit-and n (bit-not 7))]
    (loop [i (long 0)
           a0 (long 0) a1 (long 0) a2 (long 0) a3 (long 0)
           b0 (long 0) b1 (long 0) b2 (long 0) b3 (long 0)]
      (if (< i n8)
        (recur (unchecked-add i 8)
               (bit-xor a0 (extract-rgb (sm64 SEED i)))
               (bit-xor a1 (extract-rgb (sm64 SEED (unchecked-add i 1))))
               (bit-xor a2 (extract-rgb (sm64 SEED (unchecked-add i 2))))
               (bit-xor a3 (extract-rgb (sm64 SEED (unchecked-add i 3))))
               (bit-xor b0 (extract-rgb (sm64 SEED (unchecked-add i 4))))
               (bit-xor b1 (extract-rgb (sm64 SEED (unchecked-add i 5))))
               (bit-xor b2 (extract-rgb (sm64 SEED (unchecked-add i 6))))
               (bit-xor b3 (extract-rgb (sm64 SEED (unchecked-add i 7)))))
        (let [base (bit-xor a0 (bit-xor a1 (bit-xor a2 (bit-xor a3
                     (bit-xor b0 (bit-xor b1 (bit-xor b2 b3)))))))]
          (loop [j i result base]
            (if (< j n)
              (recur (unchecked-inc j) (bit-xor result (extract-rgb (sm64 SEED j))))
              result)))))))

;; L3: pmap parallel (virtual threads in bb)
(defn l3-pmap ^long [^long n ^long nthreads]
  (let [chunk (quot n nthreads)
        ranges (for [tid (range nthreads)]
                 [(long (* tid chunk))
                  (if (= tid (dec nthreads)) n (* (inc tid) chunk))])
        worker (fn [[^long start ^long end-idx]]
                 (let [count (- end-idx start)
                       n8 (bit-and count (bit-not 7))]
                   (loop [j (long 0)
                          a0 (long 0) a1 (long 0) a2 (long 0) a3 (long 0)
                          b0 (long 0) b1 (long 0) b2 (long 0) b3 (long 0)]
                     (if (< j n8)
                       (let [idx (unchecked-add start j)]
                         (recur (unchecked-add j 8)
                                (bit-xor a0 (extract-rgb (sm64 SEED idx)))
                                (bit-xor a1 (extract-rgb (sm64 SEED (unchecked-add idx 1))))
                                (bit-xor a2 (extract-rgb (sm64 SEED (unchecked-add idx 2))))
                                (bit-xor a3 (extract-rgb (sm64 SEED (unchecked-add idx 3))))
                                (bit-xor b0 (extract-rgb (sm64 SEED (unchecked-add idx 4))))
                                (bit-xor b1 (extract-rgb (sm64 SEED (unchecked-add idx 5))))
                                (bit-xor b2 (extract-rgb (sm64 SEED (unchecked-add idx 6))))
                                (bit-xor b3 (extract-rgb (sm64 SEED (unchecked-add idx 7))))))
                       (let [base (bit-xor a0 (bit-xor a1 (bit-xor a2 (bit-xor a3
                                    (bit-xor b0 (bit-xor b1 (bit-xor b2 b3)))))))]
                         (loop [k j result base]
                           (if (< k count)
                             (recur (unchecked-inc k) (bit-xor result (extract-rgb (sm64 SEED (unchecked-add start k)))))
                             result)))))))
        partials (pmap worker ranges)]
    (reduce bit-xor 0 partials)))

;; Bench harness
(defn bench [label n f]
  (f (max 1 (quot n 100))) ;; warmup
  (let [t0 (System/nanoTime)
        xor (f n)
        t1 (System/nanoTime)
        ns (- t1 t0)
        rate-m (if (zero? ns) 0 (long (/ (* n 1000.0) ns)))]
    {:label label :ns ns :n n :xor xor :rate-m rate-m}))

(defn main []
  (let [cores (.availableProcessors (Runtime/getRuntime))
        sizes [1000000 10000000]
        labels ["1M" "10M"]]

    (println)
    (println "+======================================================================+")
    (println "|       SPI VIRTUOSO — Babashka/JVM Racer                              |")
    (println "|  Tricks: unchecked-math, pmap, virtual threads, loop/recur           |")
    (println "+======================================================================+")
    (printf "  CPU cores: %d  Seed: %d%n" cores SEED)
    (println)

    ;; Single-threaded
    (let [levels [["L0" "Scalar loop/recur              " l0-scalar]
                  ["L1" "8-wide unrolled pipeline        " l1-pipeline8]
                  ["L2" "Fused gen+RGB+XOR              " l2-fused]]]
      (printf "  Level  Description                      ")
      (doseq [l labels] (printf "%13s" l))
      (println)
      (printf "  -----  -------------------------------- ")
      (doseq [_ labels] (printf " ------------"))
      (println)
      (doseq [[code desc f] levels]
        (printf "  %s    %s" code desc)
        (doseq [n sizes]
          (let [r (bench code n f)]
            (printf "%9d M/s" (:rate-m r))))
        (println)))

    ;; pmap parallel
    (println)
    (printf "  -- L3: pmap parallel (%d threads) --%n" cores)
    (printf "  L3    pmap fused parallel              ")
    (doseq [n sizes]
      (let [r (bench "L3" n #(l3-pmap % cores))]
        (printf "%9d M/s" (:rate-m r))))
    (println)

    (println)
    (println "  Tricks applied:")
    (println "    L0: loop/recur — zero allocation, tail-call optimized")
    (println "    L1: 8 independent accumulators in loop/recur bindings")
    (println "    L2: Fused gen+RGB+XOR — same pipeline as Zig/Julia/Swift")
    (println "    L3: pmap — Clojure's lazy parallel map over thread pool")
    (println)
    (println "  Compare: zig-syrup spi-virtuoso, Gay.jl spi_virtuoso.jl, swift-racer")))

(main)
