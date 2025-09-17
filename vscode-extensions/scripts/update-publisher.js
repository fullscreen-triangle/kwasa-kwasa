#!/usr/bin/env node

/**
 * Script to update publisher name in all extension package.json files
 * Usage: node scripts/update-publisher.js [publisher-name]
 */

const fs = require('fs');
const path = require('path');

const extensions = [
    'turbulance-language-server',
    'knowledge-resolution-engine',
    'pattern-evidence-workbench',
    'metacognitive-orchestration-debugger',
    'boundary-detection-studio'
];

const defaultPublisher = 'kwasa-kwasa-team';
const publisherName = process.argv[2] || defaultPublisher;

console.log(`🔧 Updating publisher to: ${publisherName}`);

extensions.forEach(extDir => {
    const packageJsonPath = path.join(__dirname, '..', extDir, 'package.json');

    if (fs.existsSync(packageJsonPath)) {
        console.log(`📁 Updating ${extDir}...`);

        // Read package.json
        const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

        // Update publisher
        packageJson.publisher = publisherName;

        // Write back to file
        fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 4) + '\n');

        console.log(`   ✅ Updated publisher to: ${publisherName}`);
    } else {
        console.log(`   ⚠️  Package.json not found: ${packageJsonPath}`);
    }
});

console.log('\n🎉 Publisher update complete!');
console.log('\n📝 Next steps:');
console.log('1. npm run login');
console.log('2. npm run publish:all');
