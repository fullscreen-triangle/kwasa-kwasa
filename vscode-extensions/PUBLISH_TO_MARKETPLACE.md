# 🚀 Publishing Kwasa-Kwasa Extensions to VSCode Marketplace

This guide will help you publish the 5 Kwasa-Kwasa extensions to the VSCode Extensions Marketplace so users can install them directly from the Extensions tab in VSCode.

## 📋 Prerequisites

### 1. Create Microsoft Azure DevOps Account

1. Go to https://dev.azure.com
2. Sign up with your Microsoft account (free)
3. Create an organization (you can skip this step initially)

### 2. Create Publisher Account

1. Visit https://marketplace.visualstudio.com/manage
2. Sign in with the same Microsoft account
3. Click "Create Publisher"
4. Fill in the details:
    - **Publisher ID**: `kwasa-kwasa-team` (must be unique)
    - **Display Name**: `Kwasa-Kwasa Team`
    - **Description**: `Advanced semantic processing extensions for VSCode`

### 3. Generate Personal Access Token (PAT)

1. In Azure DevOps, go to User Settings → Personal Access Tokens
2. Click "New Token"
3. Configure:
    - **Name**: `VSCode Extensions Publishing`
    - **Scopes**: Select `Marketplace (Publish)`
    - **Expiration**: Set to 1 year
4. **Copy and save the token securely** - you can't see it again!

## 🛠️ Setup and Publish

### Step 1: Install Publishing Tools

```bash
cd vscode-extensions
npm install -g vsce
npm install
```

### Step 2: Update Publisher (if needed)

```bash
# Use your actual publisher ID if different
npm run update-publisher kwasa-kwasa-team
```

### Step 3: Login to Marketplace

```bash
npm run login
# When prompted, paste your Personal Access Token
```

### Step 4: Build Extensions

```bash
npm run install:all
npm run compile:all
```

### Step 5: Publish to Marketplace

```bash
# Publish all extensions at once
npm run publish:all
```

Or publish individually:

```bash
npm run publish:turbulance
npm run publish:knowledge
npm run publish:pattern
npm run publish:orchestration
npm run publish:boundary
```

## ✅ After Publishing

### Verify Extensions are Live

1. Wait 5-10 minutes for propagation
2. Open VSCode
3. Go to Extensions tab (`Ctrl+Shift+X`)
4. Search for each extension:
    - "Turbulance Language Support"
    - "Knowledge Resolution Engine"
    - "Pattern Analysis Evidence Workbench"
    - "Metacognitive Orchestration Debugger"
    - "Advanced Boundary Detection Studio"

### Check Marketplace Pages

Visit: https://marketplace.visualstudio.com/publishers/kwasa-kwasa-team

You should see all 5 extensions listed with:

-   ✅ Professional descriptions
-   ✅ Keywords for searchability
-   ✅ Category classifications
-   ✅ Installation instructions
-   ✅ Repository links

## 🔄 Updating Extensions

### For Bug Fixes (Patch Updates)

```bash
npm run version:patch    # Bumps all to 0.1.1, 0.1.2, etc.
npm run publish:all
```

### For New Features (Minor Updates)

```bash
npm run version:minor    # Bumps all to 0.2.0, 0.3.0, etc.
npm run publish:all
```

### For Breaking Changes (Major Updates)

```bash
npm run version:major    # Bumps all to 1.0.0, 2.0.0, etc.
npm run publish:all
```

## 🔍 How Users Will Find Extensions

### Search Terms That Will Work:

-   **"turbulance"** → Turbulance Language Support
-   **"knowledge resolution"** → Knowledge Resolution Engine
-   **"pattern analysis"** → Pattern Analysis & Evidence Workbench
-   **"orchestration debugger"** → Metacognitive Orchestration Debugger
-   **"boundary detection"** → Advanced Boundary Detection Studio
-   **"kwasa-kwasa"** → All extensions
-   **"semantic processing"** → All extensions

### Categories:

-   **Programming Languages** (Turbulance Language Support)
-   **Data Science** (Knowledge Engine, Pattern Workbench, Boundary Studio)
-   **Debuggers** (Orchestration Debugger)
-   **Visualization** (Pattern Workbench, Boundary Studio)

## 📊 Extension Marketplace Appearance

Each extension will appear with:

### Turbulance Language Support

-   **🟦 Blue banner** (`#1e1e1e`)
-   **Keywords**: turbulance, semantic programming, language server
-   **Category**: Programming Languages

### Knowledge Resolution Engine

-   **🔵 Blue banner** (`#0066cc`)
-   **Keywords**: knowledge validation, embedding analysis, epistemic confidence
-   **Category**: Data Science

### Pattern Analysis & Evidence Workbench

-   **🟣 Purple banner** (`#6600cc`)
-   **Keywords**: pattern analysis, evidence networks, hypothesis testing
-   **Category**: Data Science

### Metacognitive Orchestration Debugger

-   **🔴 Red banner** (`#cc3300`)
-   **Keywords**: debugging, metacognitive processing, orchestration visualization
-   **Category**: Debuggers

### Advanced Boundary Detection Studio

-   **🟢 Green banner** (`#009966`)
-   **Keywords**: boundary detection, text analysis, semantic boundaries
-   **Category**: Data Science

## 🚨 Troubleshooting Publishing

### "Publisher not found" error

```bash
# Make sure you're logged in with correct publisher
vsce logout
npm run login
```

### "Package.json validation failed"

-   Check all package.json files have correct publisher name
-   Ensure version numbers are valid (semantic versioning)
-   Verify all required fields are present

### "Authentication failed"

-   Generate a new Personal Access Token
-   Ensure token has `Marketplace (Publish)` scope
-   Re-login with the new token

### "Extension already exists"

-   Use `vsce publish patch/minor/major` instead of `vsce publish`
-   Or update version manually in package.json

## 🎯 Post-Publishing Checklist

-   [ ] ✅ All 5 extensions appear in marketplace
-   [ ] ✅ Extensions install correctly from marketplace
-   [ ] ✅ Search terms find the extensions
-   [ ] ✅ Extension descriptions are clear and professional
-   [ ] ✅ All marketplace links work
-   [ ] ✅ Extensions activate properly after marketplace install
-   [ ] ✅ No broken functionality compared to local install

## 📈 Promotion

Once published, share the extensions:

### Documentation Updates

-   Update main README with marketplace installation instructions
-   Add marketplace badges to documentation
-   Create installation video tutorials

### Community Sharing

-   Announce on semantic processing forums
-   Share in VSCode extension communities
-   Create blog posts about the extensions
-   Submit to awesome-vscode lists

## 🔐 Security Note

**Never commit your Personal Access Token to git!**

For automated publishing via GitHub Actions:

1. Add PAT as repository secret: `VSCE_PAT`
2. Use the provided GitHub Actions workflow
3. Tag releases to trigger automatic publishing

---

**🎉 Once published, users can simply search "kwasa-kwasa" in the Extensions tab and install all 5 extensions with one click each!**
