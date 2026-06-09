/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config) => {
    // @xenova/transformers references node-only modules (onnxruntime-node, sharp);
    // don't bundle them for the browser build.
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };
    return config;
  },
};

module.exports = nextConfig;
