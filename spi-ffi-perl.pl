#!/usr/bin/env perl
# SPI FFI from Perl via FFI::Platypus (fallback: inline dlopen)
use strict;
use warnings;
use Time::HiRes qw(clock_gettime CLOCK_MONOTONIC);
use Config;

my $lib = "$FindBin::Bin/zig-out/lib/libspi.dylib" if eval { require FindBin; 1 };
$lib //= do { my $d = $0; $d =~ s|[^/]*$||; "${d}zig-out/lib/libspi.dylib" };

# Use DynaLoader directly
require DynaLoader;
my $handle = DynaLoader::dl_load_file($lib, 0) or die "Cannot load $lib: " . DynaLoader::dl_error();

my $color_at_sym = DynaLoader::dl_find_symbol($handle, "spi_color_at") or die "no spi_color_at";
my $trit_sym = DynaLoader::dl_find_symbol($handle, "spi_trit") or die "no spi_trit";
my $fp_sym = DynaLoader::dl_find_symbol($handle, "spi_xor_fingerprint") or die "no spi_xor_fingerprint";
my $fpp_sym = DynaLoader::dl_find_symbol($handle, "spi_xor_fingerprint_parallel") or die "no spi_xor_fingerprint_parallel";

# Perl can't easily call C functions without XS/FFI module — just prove the symbols resolve
printf "Perl %vd: loaded libspi.dylib OK\n", $^V;
printf "  symbols: color_at=%s trit=%s fp=%s fp_parallel=%s\n",
    $color_at_sym ? "OK" : "FAIL",
    $trit_sym ? "OK" : "FAIL",
    $fp_sym ? "OK" : "FAIL",
    $fpp_sym ? "OK" : "FAIL";
print "  (Perl DynaLoader resolves symbols; calling requires FFI::Platypus or Inline::C)\n";
