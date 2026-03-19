#!/usr/bin/env ruby
# SPI FFI from Ruby via fiddle
require 'fiddle'
require 'fiddle/import'

module SPI
  extend Fiddle::Importer
  dlload File.join(__dir__, 'zig-out', 'lib', 'libspi.dylib')
  extern 'unsigned long long spi_color_at(unsigned long long, unsigned long long)'
  extern 'signed char spi_trit(unsigned long long, unsigned long long)'
  extern 'unsigned long long spi_xor_fingerprint(unsigned long long, unsigned long long, unsigned long long)'
  extern 'unsigned long long spi_xor_fingerprint_parallel(unsigned long long, unsigned long long, unsigned int)'
end

SEED = 42
puts "Ruby #{RUBY_VERSION}: color_at(42,0)=#{'%06x' % SPI.spi_color_at(SEED,0)} " \
     "color_at(42,69)=#{'%06x' % SPI.spi_color_at(SEED,69)} " \
     "trit(42,0)=#{SPI.spi_trit(SEED,0)} trit(42,69)=#{SPI.spi_trit(SEED,69)}"

[1_000_000, 100_000_000].each do |n|
  SPI.spi_xor_fingerprint_parallel(SEED, 1000, 0)
  t = Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)
  xor = SPI.spi_xor_fingerprint_parallel(SEED, n, 0)
  ns = Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond) - t
  rate = n * 1000 / ns
  puts "  #{n/1_000_000}M: #{rate} M/s  xor=0x#{'%012x' % xor}"
end
