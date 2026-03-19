#!/usr/bin/env guile
!#
;;; SPI FFI from GNU Guile Scheme via foreign function interface
(use-modules (system foreign)
             (system foreign-library)
             (ice-9 format))

(define libspi (load-foreign-library
                (string-append (dirname (current-filename))
                               "/zig-out/lib/libspi")))

(define spi-color-at
  (foreign-library-function libspi "spi_color_at"
    #:return-type uint64 #:arg-types (list uint64 uint64)))

(define spi-trit
  (foreign-library-function libspi "spi_trit"
    #:return-type int8 #:arg-types (list uint64 uint64)))

(define spi-xor-fingerprint
  (foreign-library-function libspi "spi_xor_fingerprint"
    #:return-type uint64 #:arg-types (list uint64 uint64 uint64)))

(define spi-xor-fingerprint-parallel
  (foreign-library-function libspi "spi_xor_fingerprint_parallel"
    #:return-type uint64 #:arg-types (list uint64 uint64 uint32)))

(define SEED 42)

(format #t "Guile ~a: color_at(42,0)=#~6,'0x color_at(42,69)=#~6,'0x~%"
        (version)
        (spi-color-at SEED 0)
        (spi-color-at SEED 69))
(format #t "  trit(42,0)=~a trit(42,69)=~a~%"
        (spi-trit SEED 0) (spi-trit SEED 69))

(define (bench label n)
  (spi-xor-fingerprint-parallel SEED 1000 0) ; warmup
  (let* ((t0 (get-internal-real-time))
         (xor (spi-xor-fingerprint-parallel SEED n 0))
         (t1 (get-internal-real-time))
         (ns (* (- t1 t0) (/ 1000000000 internal-time-units-per-second)))
         (rate (if (> ns 0) (quotient (* n 1000) ns) 0)))
    (format #t "  ~a: ~a M/s  xor=0x~12,'0x~%" label rate xor)))

(bench "1M" 1000000)
(bench "100M" 100000000)
(bench "1B" 1000000000)
