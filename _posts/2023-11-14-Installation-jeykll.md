---
layout: post
title:  "Downloading and Running a Jekyll Site Locally on Windows Using VSCode"
date:   2023-11-14 5:31:29 +0900
categories: Update
---
### Downloading and Running a Jekyll Site Locally on Windows Using VSCode
We will cover in this section:
1. **Setting Up Prerequisites**: Installing Ruby, Jekyll, Git, and VSCode on Windows.
2. **Cloning the GitHub Repository**: Using Git to download the Jekyll site code from GitHub.
3. **Installing Dependencies**: Using `bundle install` to set up necessary gems for Jekyll.
4. **Serving the Site Locally**: Running `bundle exec jekyll serve` to start the local Jekyll server and preview the site.

To download your GitHub repository and run Jekyll on Visual Studio Code (VSCode) on Windows, follow these steps:

### Step 1: Install Prerequisites
1. **Install Ruby and Jekyll**:
   - Download and install Ruby for Windows from the [RubyInstaller](https://rubyinstaller.org/).
   - After installation, open a new terminal (Command Prompt or PowerShell) and install Jekyll and Bundler by running:
     ```bash
     gem install jekyll bundler
     ```

2. **Install Git**:
   - Download and install Git for Windows from [git-scm.com](https://git-scm.com/). This will allow you to clone your repository.

3. **Install VSCode**:
   - Download and install [Visual Studio Code](https://code.visualstudio.com/) if you haven’t already.

### Step 2: Clone Your GitHub Repository
1. **Open Git Bash or Command Prompt**:
   - In Windows, open Git Bash, Command Prompt, or PowerShell.

2. **Navigate to Your Desired Directory**:
   ```bash
   cd path\to\your\desired\directory
   ```

3. **Clone the Repository**:
   - Replace `your-username` and `your-repository` with your GitHub username and repository name:
     ```bash
     git clone https://github.com/your-username/your-repository.git
     ```
   - This will download the repository to your local directory.

### Step 3: Open the Project in VSCode
1. Open Visual Studio Code and go to `File` > `Open Folder...`.
2. Select the folder where you cloned your GitHub repository.

### Step 4: Install Jekyll Dependencies
1. **Open the Terminal in VSCode**:
   - Go to `View` > `Terminal` to open the integrated terminal in VSCode.

2. **Navigate to Your Jekyll Project Directory**:
   - Ensure you’re in the correct directory where your Jekyll project is located.

3. **Install Dependencies with Bundler**:
   - Run the following command to install the necessary dependencies:
     ```bash
     bundle install
     ```

### Step 5: Serve the Jekyll Site Locally
1. **Run the Jekyll Server**:
   - Start the Jekyll server by running:
     ```bash
     bundle exec jekyll serve
     ```
   - This will start a local server for your Jekyll site. You can access it at `http://127.0.0.1:4000` in your browser.

2. **View Live Changes**:
   - As you make updates in VSCode, Jekyll will automatically regenerate the site, allowing you to see changes immediately in your browser.

### Additional Tips
- **Incremental Builds**: For faster builds during development, you can add the `--incremental` flag:
  ```bash
  bundle exec jekyll serve --incremental
  ```
- **Error Debugging**: If you encounter any errors, try using `--trace` for detailed output:
  ```bash
  bundle exec jekyll serve --trace
  ```


