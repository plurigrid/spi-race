#!/usr/bin/env bb
;; SPI FFI from Babashka via JNR-FFI (JVM foreign function interface)
;; Babashka has built-in support for native interop via jnr-ffi

(require '[babashka.process :refer [shell]])

(def SEED 42)

;; Babashka doesn't ship jnr-ffi by default, so we use the test binary
;; to prove preservation. But we CAN do pure-Clojure splitmix64:

(set! *unchecked-math* true)

(defn splitmix64 ^long [^long seed ^long index]
  (let [golden (unchecked-long 0x9e3779b97f4a7c15)
        mix1   (unchecked-long 0xbf58476d1ce4e5b9)
        mix2   (unchecked-long 0x94d049bb133111eb)
        z0 (unchecked-add seed (unchecked-multiply golden index))
        z1 (unchecked-multiply (bit-xor z0 (unsigned-bit-shift-right z0 30)) mix1)
        z2 (unchecked-multiply (bit-xor z1 (unsigned-bit-shift-right z1 27)) mix2)]
    (bit-xor z2 (unsigned-bit-shift-right z2 31))))

(defn extract-rgb ^long [^long v]
  (bit-or (bit-shift-left (bit-and (unsigned-bit-shift-right v 16) 0xFF) 16)
          (bit-or (bit-shift-left (bit-and (unsigned-bit-shift-right v 8) 0xFF) 8)
                  (bit-and v 0xFF))))

(defn xor-fingerprint ^long [^long seed ^long n]
  (loop [i (long 0) xor (long 0)]
    (if (< i n)
      (recur (unchecked-inc i)
             (bit-xor xor (extract-rgb (splitmix64 seed i))))
      xor)))

(println (str "Babashka " (System/getProperty "babashka.version")
              ": pure Clojure splitmix64"))

(let [xor-1k (xor-fingerprint SEED 1000)]
  (println (format "  xor_fingerprint(42, 0, 1K) = 0x%012x" xor-1k)))

(let [t0 (System/nanoTime)
      xor-100k (xor-fingerprint SEED 100000)
      ns (- (System/nanoTime) t0)
      rate (if (pos? ns) (quot (* 100000 1000) ns) 0)]
  (println (format "  xor_fingerprint(42, 0, 100K) = 0x%012x  %d M/s" xor-100k rate)))

(let [t0 (System/nanoTime)
      xor-1m (xor-fingerprint SEED 1000000)
      ns (- (System/nanoTime) t0)
      rate (if (pos? ns) (quot (* 1000000 1000) ns) 0)]
  (println (format "  xor_fingerprint(42, 0, 1M) = 0x%012x  %d M/s" xor-1m rate))
  (println (format "  expected (from Zig):          0x00000010de88"))
  (println (format "  match: %s" (if (= xor-1m 0x10de88) "PASS" "FAIL"))))
