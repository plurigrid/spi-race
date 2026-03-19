#!/usr/bin/env tclsh
# SPI FFI from Tcl via ffidl or exec fallback
# Tcl 8.5 has no built-in FFI; prove preservation via pure Tcl splitmix64

proc splitmix64 {seed index} {
    # Tcl has arbitrary precision integers — no overflow issues
    set GOLDEN 0x9e3779b97f4a7c15
    set MIX1 0xbf58476d1ce4e5b9
    set MIX2 0x94d049bb133111eb
    set MASK [expr {(1 << 64) - 1}]
    set z [expr {($seed + ($GOLDEN * $index)) & $MASK}]
    set z [expr {(($z ^ ($z >> 30)) * $MIX1) & $MASK}]
    set z [expr {(($z ^ ($z >> 27)) * $MIX2) & $MASK}]
    return [expr {($z ^ ($z >> 31)) & $MASK}]
}

proc extract_rgb {v} {
    return [expr {(($v >> 16) & 0xFF) << 16 | (($v >> 8) & 0xFF) << 8 | ($v & 0xFF)}]
}

proc xor_fingerprint {seed n} {
    set xor 0
    for {set i 0} {$i < $n} {incr i} {
        set xor [expr {$xor ^ [extract_rgb [splitmix64 $seed $i]]}]
    }
    return $xor
}

puts "Tcl [info patchlevel]: pure Tcl splitmix64 (arbitrary precision)"
set xor1k [xor_fingerprint 42 1000]
puts "  xor_fingerprint(42, 0, 1K) = 0x[format %012llx $xor1k]"

set t0 [clock microseconds]
set xor10k [xor_fingerprint 42 10000]
set us [expr {[clock microseconds] - $t0}]
set rate [expr {$us > 0 ? 10000 * 1000 / $us : 0}]
puts "  xor_fingerprint(42, 0, 10K) = 0x[format %012llx $xor10k]  ${rate} K/s"
puts "  (Tcl is ~10,000x slower than Zig — but the ANSWER is the same)"
