module.exports = {
  reactStrictMode: false,
  webpack: (config) => {
    config.module.rules.push({
      test: /\.glb$/,
      type: 'asset/resource',
    })
    return config
  },
}
