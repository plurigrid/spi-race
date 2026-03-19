#!/usr/bin/env node
// SPI preservation proof from Node.js — pure JS BigInt splitmix64
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const SEED = 42n;
const testBin = join(__dirname, 'zig-out', 'bin', 'spi-test');

// Alternatively: pure JS splitmix64 to cross-verify
function splitmix64(seed, index) {
  const GOLDEN = 0x9e3779b97f4a7c15n;
  const MIX1 = 0xbf58476d1ce4e5b9n;
  const MIX2 = 0x94d049bb133111ebn;
  const MASK = (1n << 64n) - 1n;
  let z = (seed + ((GOLDEN * index) & MASK)) & MASK;
  z = (((z ^ (z >> 30n)) & MASK) * MIX1) & MASK;
  z = (((z ^ (z >> 27n)) & MASK) * MIX2) & MASK;
  return (z ^ (z >> 31n)) & MASK;
}

function extractRgb(v) {
  return ((v >> 16n) & 0xFFn) << 16n | ((v >> 8n) & 0xFFn) << 8n | (v & 0xFFn);
}

function xorFingerprint(seed, n) {
  let xor = 0n;
  for (let i = 0n; i < n; i++) {
    xor ^= extractRgb(splitmix64(seed, i));
  }
  return xor;
}

// Verify JS matches Zig for small N
const jsXor1M = xorFingerprint(SEED, 1000000n);
console.log(`Node.js ${process.version}: pure JS splitmix64`);
console.log(`  xor_fingerprint(42, 0, 1M) = 0x${jsXor1M.toString(16).padStart(12, '0')}`);
console.log(`  expected (from Zig):          0x00000010de88`);
console.log(`  match: ${jsXor1M === 0x10de88n ? 'PASS' : 'FAIL'}`);

// Benchmark JS (will be slow — BigInt overhead)
const t0 = process.hrtime.bigint();
const jsXor100k = xorFingerprint(SEED, 100000n);
const ns = process.hrtime.bigint() - t0;
const rate = Number(100000n * 1000n / ns);
console.log(`  100K: ${rate} M/s (BigInt overhead ~200x vs native)`);
