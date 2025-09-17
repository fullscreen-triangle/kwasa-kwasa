# 📦 Publishing to VSCode Extensions Marketplace

This guide explains how to publish the Kwasa-Kwasa extensions to the VSCode Extensions Marketplace so users can install them directly from the Extensions tab.

## 🚀 Quick Publishing Steps

### Prerequisites

1. **Microsoft Azure DevOps Account** (free)

    - Go to https://dev.azure.com
    - Sign up with Microsoft account

2. **Personal Access Token (PAT)**

    - In Azure DevOps: User Settings > Personal Access Tokens
    - Create token with **Marketplace (publish)** scope
    - Save the token securely

3. **Publisher Account**
    - Go to https://marketplace.visualstudio.com/manage
    - Create a publisher account
    - Choose a unique publisher ID (e.g., `kwasa-kwasa-team`)

### Setup Publishing

1. **Install vsce globally:**

```bash
npm install -g vsce
```

2. **Login with your PAT:**

```bash
vsce login your-publisher-name
# Enter your Personal Access Token when prompted
```

3. **Update publisher in all package.json files:**

```bash
# Run this script to update all extensions
npm run update-publisher
```

4. **Publish all extensions:**

```bash
npm run publish:all
```

## 🔍 After Publishing

Users will be able to find your extensions by searching in VSCode:

1. **Open Extensions tab** (`Ctrl+Shift+X`)
2. **Search for:**
    - "Turbulance Language Server"
    - "Knowledge Resolution Engine"
    - "Pattern Evidence Workbench"
    - "Metacognitive Orchestration Debugger"
    - "Boundary Detection Studio"
3. **Click Install** on each extension
4. **Reload VSCode** when prompted

## 📝 Extension Marketplace Information

Each extension will appear with:

-   **Professional descriptions**
-   **Feature highlights**
-   **Screenshots** (when added)
-   **Usage instructions**
-   **Keywords for discoverability**
-   **Links to documentation**

## 🔄 Updating Extensions

To publish updates:

```bash
# Update version and publish
npm run version:patch  # or version:minor, version:major
npm run publish:all
```

## 🏷️ Extension Names in Marketplace

-   **turbulance-language-server** → "Turbulance Language Support"
-   **knowledge-resolution-engine** → "Knowledge Resolution Engine"
-   **pattern-evidence-workbench** → "Pattern Analysis & Evidence Workbench"
-   **metacognitive-orchestration-debugger** → "Metacognitive Orchestration Debugger"
-   **boundary-detection-studio** → "Advanced Boundary Detection Studio"

## 🔍 Search Keywords

Users can find extensions by searching:

-   "turbulance"
-   "semantic processing"
-   "knowledge validation"
-   "pattern analysis"
-   "boundary detection"
-   "kwasa-kwasa"
-   "evidence-based reasoning"
-   "metacognitive debugging"
