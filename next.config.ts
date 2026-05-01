import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Keep onnxruntime-node out of the webpack bundle — it ships native .node binaries
  serverExternalPackages: ['onnxruntime-node'],
}

export default nextConfig
