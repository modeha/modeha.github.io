---
layout: post
title: essential Git commands for managing your GitHub Pages repository
date: 2024-11-14 15:54 -0500
---
Here’s a list of essential Git commands for managing a GitHub Pages repository I will consider  (`modeha.github.io`), including creating branches, committing changes, merging, and pushing to GitHub.

### Initial Setup

1. **Clone the Repository** (if not already cloned):
   ```bash
   git clone https://github.com/your-username/modeha.github.io.git
   cd modeha.github.io
   ```

2. **Check the Status**:
   - To check the current status of your working directory and see any changes:
     ```bash
     git status
     ```

### Basic Workflow Commands

1. **Add Files to Staging**:
   - Adds all changed files to the staging area:
     ```bash
     git add .
     ```
   - Or, add a specific file:
     ```bash
     git add filename
     ```

2. **Commit Changes**:
   - Commit staged changes with a descriptive message:
     ```bash
     git commit -m "Your commit message"
     ```

3. **Push Changes to GitHub**:
   - Push commits to the main branch:
     ```bash
     git push origin main
     ```

### Working with Branches

1. **Create a New Branch**:
   - Create and switch to a new branch:
     ```bash
     git checkout -b new-branch-name
     ```

2. **Switch to an Existing Branch**:
   ```bash
   git checkout branch-name
   ```

3. **Push a New Branch to GitHub**:
   - After creating a new branch locally, push it to GitHub:
     ```bash
     git push -u origin new-branch-name
     ```

4. **List All Branches**:
   - To see all branches (local and remote):
     ```bash
     git branch -a
     ```

### Merging Branches

1. **Merge a Branch into Main**:
   - First, switch to the main branch:
     ```bash
     git checkout main
     ```
   - Pull the latest changes from GitHub to ensure it’s up to date:
     ```bash
     git pull origin main
     ```
   - Merge the feature branch into main:
     ```bash
     git merge feature-branch-name
     ```

2. **Resolve Merge Conflicts** (if any):
   - Open the conflicting files, make necessary edits, then add the resolved files:
     ```bash
     git add resolved-file
     ```
   - Commit the merge after resolving conflicts:
     ```bash
     git commit -m "Resolved merge conflicts"
     ```

3. **Delete a Merged Branch**:
   - Delete the branch locally after it’s merged:
     ```bash
     git branch -d feature-branch-name
     ```
   - Delete the branch from GitHub:
     ```bash
     git push origin --delete feature-branch-name
     ```

### Syncing with Remote

1. **Pull Changes from GitHub**:
   - To sync the local branch with changes from GitHub:
     ```bash
     git pull origin branch-name
     ```

2. **Fetch Remote Changes Without Merging**:
   - Downloads changes from GitHub but doesn’t merge them into your current branch:
     ```bash
     git fetch origin
     ```

### Undoing Changes

1. **Undo Changes in a File**:
   - Revert changes in a file that hasn’t been staged:
     ```bash
     git checkout -- filename
     ```

2. **Unstage Changes**:
   - If you added changes with `git add` but want to remove them from the staging area:
     ```bash
     git reset filename
     ```

3. **Revert a Commit**:
   - To undo the latest commit but keep the changes in your working directory:
     ```bash
     git reset --soft HEAD~1
     ```

### Pushing to GitHub Pages

If you’re using GitHub Pages, pushing to the `main` branch (or `gh-pages`, depending on your repository settings) will automatically update the site. Ensure all changes are committed and pushed to the branch that GitHub Pages is configured to use.

1. **Push to Update GitHub Pages**:
   - If your GitHub Pages site is served from the `main` branch:
     ```bash
     git push origin main
     ```
   - If your GitHub Pages site is served from a different branch, push to that branch instead.

---

Here are some additional, less common Git commands and concepts that can be useful in specific scenarios. These commands extend beyond the basics covered above, offering more control over branches, commits, history, and repository management.

### Advanced Git Commands and Concepts

1. **Rebasing**:
   - **Rebase a Branch**: Rebasing is useful for keeping a clean, linear history by applying your changes on top of another branch’s latest changes.
     ```bash
     git checkout feature-branch
     git rebase main
     ```
   - **Interactive Rebase**: Allows you to rewrite commit history, such as squashing multiple commits into one.
     ```bash
     git rebase -i HEAD~3
     ```
     - Replace `HEAD~3` with the number of commits you want to go back.

2. **Stashing Changes**:
   - **Stash Uncommitted Changes**: Temporarily saves changes without committing them, allowing you to work on something else.
     ```bash
     git stash
     ```
   - **Apply Stashed Changes**: Reapplies the stashed changes.
     ```bash
     git stash apply
     ```
   - **List Stashes**: Shows all stashed changes.
     ```bash
     git stash list
     ```

3. **Resetting Commits**:
   - **Soft Reset**: Moves the latest commit to the staging area, keeping changes in your working directory.
     ```bash
     git reset --soft HEAD~1
     ```
   - **Hard Reset**: Deletes the latest commit and all changes. This cannot be undone, so use with caution.
     ```bash
     git reset --hard HEAD~1
     ```

4. **Cherry-Picking Commits**:
   - **Cherry-Pick a Commit**: Copy a specific commit from one branch to another without merging the full branch.
     ```bash
     git cherry-pick <commit-hash>
     ```

5. **Viewing History and Differences**:
   - **View Commit Log**: Shows the commit history with messages.
     ```bash
     git log
     ```
   - **Graph Log**: Visualizes the branch and merge history.
     ```bash
     git log --oneline --graph --all --decorate
     ```
   - **Difference Between Commits**: Shows the differences between two commits.
     ```bash
     git diff <commit1> <commit2>
     ```

6. **Tagging**:
   - **Create a Tag**: Tags are useful for marking releases or specific points in the project’s history.
     ```bash
     git tag -a v1.0 -m "Version 1.0 release"
     ```
   - **Push Tags to GitHub**:
     ```bash
     git push origin v1.0
     ```
   - **List All Tags**:
     ```bash
     git tag
     ```

7. **Forking and Pull Requests**:
   - **Fork a Repository**: Create a copy of another user’s repository on GitHub to your account (done on GitHub’s web interface).
   - **Submit a Pull Request**: After pushing changes to your fork, open a pull request on GitHub to propose changes to the original repository.

8. **Git Aliases**:
   - **Create Git Aliases**: Shorten common commands. For example:
     ```bash
     git config --global alias.co checkout
     git config --global alias.br branch
     git config --global alias.ci commit
     ```

9. **Amending Commits**:
   - **Amend the Last Commit**: Add more changes to the most recent commit without creating a new one.
     ```bash
     git add .
     git commit --amend
     ```

10. **Fork Synchronization**:
    - **Update Your Fork with Upstream Changes**: Useful if you’ve forked a repository and need to sync it with changes made to the original.
      ```bash
      git remote add upstream https://github.com/original-owner/repo.git
      git fetch upstream
      git merge upstream/main
      ```

11. **Submodules**:
    - **Add a Submodule**: Add another Git repository within your current repository (useful for projects with dependencies).
      ```bash
      git submodule add https://github.com/username/another-repo.git
      ```
    - **Update Submodules**:
      ```bash
      git submodule update --init --recursive
      ```

12. **Git Clean**:
    - **Remove Untracked Files**: Deletes untracked files in the working directory. Use with caution.
      ```bash
      git clean -f
      ```
    - **Preview Clean**: See which files would be deleted without deleting them.
      ```bash
      git clean -n
      ```

These additional commands cover more advanced usage and specific workflows in Git, enhancing flexibility and control in managing your repository. Let me know if you’d like more detailed explanations or examples for any of these!

There are a few more advanced Git concepts and commands that I didn’t mention in the previous responses. These cover some niche and power-user functionalities of Git, which can be helpful in certain scenarios, especially for large projects or complex workflows.

### Additional Advanced Git Commands and Concepts

1. **Git Bisect**:
   - **Finding a Bug with Git Bisect**: A powerful feature for debugging, `git bisect` helps identify the specific commit that introduced a bug by using a binary search approach.
     ```bash
     git bisect start
     git bisect bad # Mark the current commit as bad
     git bisect good <commit-hash> # Mark an earlier known good commit
     ```
   - Git will check out each commit in the middle of the range, and you can mark each one as "good" or "bad" until you narrow down the offending commit.

2. **Git Hooks**:
   - **Automate Actions with Git Hooks**: Git hooks are custom scripts that run automatically at certain points in Git’s execution, like pre-commit, pre-push, or post-merge.
   - **Example**: A pre-commit hook to format code automatically:
     ```bash
     echo 'black .' > .git/hooks/pre-commit
     chmod +x .git/hooks/pre-commit
     ```
   - Common hooks include:
     - `pre-commit`: Runs before committing changes.
     - `pre-push`: Runs before pushing commits.
     - `post-merge`: Runs after merging.

3. **Squashing Commits**:
   - **Squash Commits into One**: Useful for cleaning up commit history before merging, especially in pull requests.
     - Interactive rebase for squashing:
       ```bash
       git rebase -i HEAD~3
       ```
     - Mark commits with `squash` to combine them into one.

4. **Git Blame**:
   - **Check Who Changed a Line Last**: `git blame` is useful for finding out who last modified each line in a file.
     ```bash
     git blame filename
     ```

5. **Git Reflog**:
   - **Recover Lost Commits with Git Reflog**: If you’ve made changes and can’t find them in your branch history, `git reflog` can help recover lost commits.
     ```bash
     git reflog
     ```
   - It shows a log of changes made in the repository, including detached heads or rebases.

6. **Detach Head**:
   - **Checkout a Specific Commit Without Changing the Branch**:
     ```bash
     git checkout <commit-hash>
     ```
   - This “detached HEAD” state is useful for testing or viewing an old commit without modifying the branch’s tip.

7. **Sparse-Checkout**:
   - **Check Out Only Part of a Repository**: Useful for very large repositories where you only need specific directories.
     ```bash
     git sparse-checkout init
     git sparse-checkout set path/to/directory
     ```

8. **Tracking Files with `.gitattributes`**:
   - **Customize Git Attributes for Files**: Control how Git handles certain files, like managing line endings or enabling merge strategies.
   - Example `.gitattributes`:
     ```text
     *.md text
     *.jpg binary
     ```

9. **Git Diff Options**:
   - **Show Word-Level Changes**:
     ```bash
     git diff --word-diff
     ```
   - **View Stat Summary**:
     ```bash
     git diff --stat
     ```

10. **Pruning Stale Branches**:
    - **Remove Local References to Deleted Remote Branches**:
      ```bash
      git remote prune origin
      ```

11. **Git Worktrees**:
    - **Multiple Working Directories in One Repository**: Allows you to work on multiple branches simultaneously without cloning the repository multiple times.
      ```bash
      git worktree add ../path-to-new-worktree branch-name
      ```

12. **Git Archive**:
    - **Export a Specific Branch or Commit as a ZIP File**:
      ```bash
      git archive -o archive.zip HEAD
      ```

13. **Replace Commits with Filter-Branch (or BFG)**:
    - **Rewrite History to Remove Large or Sensitive Files**:
      ```bash
      git filter-branch --tree-filter 'rm -f path/to/file' HEAD
      ```
    - For a safer and faster approach, use [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/).

14. **Managing Subtrees**:
    - **Add a Repository as a Subtree**: Useful for embedding an entire repository as a subfolder within another repository.
      ```bash
      git subtree add --prefix=subfolder-name https://github.com/username/repo.git branch-name
      ```
    - **Update a Subtree**:
      ```bash
      git subtree pull --prefix=subfolder-name https://github.com/username/repo.git branch-name
      ```

15. **Setting Up a Bare Repository**:
    - **Create a Shared Repository Without Working Directory**:
      ```bash
      git init --bare
      ```
    - Useful for setting up a central Git repository to push and pull from.

### Summary of Advanced Git Functionalities

These advanced Git commands cover areas like **history rewriting**, **branch optimization**, **project modularization**, and **automated workflow management**. While some commands are more specialized, understanding them can give you more flexibility in managing and refining your Git workflow, especially in collaborative and large-scale projects.

To dive deeper into any specific Git command or concept you'd like to explore further. Here are a few commands and concepts from the list above that often benefit from detailed explanations and examples:

1. **Git Bisect** - Finding a bug through binary search within commit history.
2. **Interactive Rebase** - Squashing, reordering, and editing commits to clean up history.
3. **Git Reflog** - Recovering lost commits or branches.
4. **Git Hooks** - Automating workflows with pre-commit, pre-push, etc.
5. **Git Worktrees** - Managing multiple working directories within a single repository.
6. **Sparse Checkout** - Cloning only specific parts of a large repository.
7. **Filter-Branch and BFG Repo-Cleaner** - Removing large or sensitive data from history.

Here’s a detailed explanation of each of the advanced Git commands and concepts.

---

### 1. **Git Bisect** - Finding a Bug with Binary Search

Git Bisect is a powerful tool for finding the exact commit where a bug was introduced. It works by performing a binary search on the commit history.

#### How It Works:
1. Start bisect:
   ```bash
   git bisect start
   ```

2. Mark the current commit as "bad" (i.e., it contains the bug):
   ```bash
   git bisect bad
   ```

3. Mark an older commit as "good" (a commit where the bug did not exist):
   ```bash
   git bisect good <commit-hash>
   ```

4. Git will now checkout a commit halfway between the "good" and "bad" commits. Test this commit, and mark it as either "good" or "bad":
   ```bash
   git bisect good  # or git bisect bad
   ```

5. Repeat until Git identifies the exact commit where the bug was introduced.

6. To exit bisect mode, use:
   ```bash
   git bisect reset
   ```

---

### 2. **Interactive Rebase** - Rewriting Commit History

Interactive rebasing allows you to modify, reorder, squash, and delete commits to clean up your commit history. It’s especially useful before merging a feature branch into the main branch.

#### Basic Steps:
1. Start an interactive rebase on the last few commits (e.g., 3 commits):
   ```bash
   git rebase -i HEAD~3
   ```

2. This opens an editor showing the last 3 commits. Each line represents a commit and starts with a command (e.g., `pick`). Available commands:
   - **pick**: Keep the commit as-is.
   - **reword**: Keep the commit but change the commit message.
   - **edit**: Pause the rebase to make changes to this commit.
   - **squash (s)**: Combine this commit with the previous one.
   - **drop (d)**: Delete the commit.

3. Save and close the editor to apply changes.

4. If you chose to **edit** a commit, make the changes and use:
   ```bash
   git commit --amend
   git rebase --continue
   ```

---

### 3. **Git Reflog** - Recovering Lost Commits or Branches

Git Reflog (Reference Log) records changes to the tip of branches, allowing you to recover lost commits or branches.

#### Usage:
1. View the reflog:
   ```bash
   git reflog
   ```

2. The output shows recent actions along with commit hashes. If you see a commit you want to recover, you can checkout or reset to it:
   ```bash
   git checkout <commit-hash>
   ```

3. **Example**: If you accidentally deleted a branch and want to recover its latest commit, use `git reflog` to find the commit hash, then create a new branch from it:
   ```bash
   git branch recovered-branch <commit-hash>
   ```

---

### 4. **Git Hooks** - Automating Workflows

Git Hooks are scripts that automatically run at certain points in your Git workflow, like pre-commit or pre-push. They’re useful for enforcing code style, running tests, or automating tasks.

#### Common Hooks:
- **Pre-commit**: Runs before `git commit`, useful for code formatting.
- **Pre-push**: Runs before `git push`, useful for running tests.

#### Setting Up a Hook:
1. Go to the `.git/hooks` directory.
2. Rename or create a file for the desired hook (e.g., `pre-commit`).
3. Add a script to the file. For example, to format code before committing:
   ```bash
   #!/bin/sh
   black .
   ```
4. Make the hook executable:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

---

### 5. **Git Worktrees** - Managing Multiple Working Directories

Git Worktrees allow you to check out multiple branches at the same time in separate working directories, which is helpful for working on multiple features or versions concurrently.

#### Basic Commands:
1. Add a new worktree:
   ```bash
   git worktree add ../new-worktree-dir branch-name
   ```

2. This creates a new directory with the specified branch checked out. You can make changes independently in each worktree.

3. To remove a worktree:
   ```bash
   git worktree remove ../new-worktree-dir
   ```

---

### 6. **Sparse Checkout** - Cloning Only Specific Parts of a Repository

Sparse Checkout is useful for very large repositories where you only need certain directories, saving disk space and reducing setup time.

#### Steps to Use Sparse Checkout:
1. Enable sparse checkout in the repository:
   ```bash
   git sparse-checkout init
   ```

2. Specify the directories you want to check out:
   ```bash
   git sparse-checkout set path/to/directory
   ```

3. Git will check out only the specified directory, and you can work with it like any other repository.

---

### 7. **Filter-Branch and BFG Repo-Cleaner** - Removing Large or Sensitive Data from History

Both `git filter-branch` and **BFG Repo-Cleaner** can be used to rewrite commit history, typically to remove large files or sensitive data.

#### Using `git filter-branch`:
1. Remove a specific file from history:
   ```bash
   git filter-branch --tree-filter 'rm -f path/to/file' HEAD
   ```

2. **Important**: This rewrites history, so all branches will need to be forced pushed. It’s best used on private repos or before sharing the repository.

#### Using BFG Repo-Cleaner:
BFG is faster and simpler for large repositories. It’s a separate tool you’ll need to install (instructions [here](https://rtyley.github.io/bfg-repo-cleaner/)).

1. Remove a file or directory:
   ```bash
   bfg --delete-files path/to/file
   ```

2. **Remove All Commits with Sensitive Data** (e.g., passwords):
   ```bash
   bfg --delete-files "*.txt"
   ```

3. After running BFG, clean up and force-push:
   ```bash
   git reflog expire --expire=now --all && git gc --prune=now --aggressive
   git push --force
   ```

---

Each of these advanced commands can greatly enhance your Git workflow, especially in large or complex projects where history management, parallel development, and automation become crucial. 

There are even more advanced Git commands and techniques that can be useful in specific scenarios. Here’s a list of additional Git commands that might come in handy:

### 1. **Git Cherry** - Identifying Unique Commits

   - **Git Cherry** shows which commits exist on your current branch that are not in another branch. It’s useful for checking which changes are unique to a branch.
     ```bash
     git cherry <upstream-branch>
     ```

### 2. **Git Shortlog** - Summarize Commit History

   - **Git Shortlog** organizes commit history by author and displays the commit messages in a summarized format, which is useful for generating changelogs.
     ```bash
     git shortlog
     ```

### 3. **Git Describe** - Naming Commits Based on Tags

   - **Git Describe** outputs a name for the current commit based on the most recent tag. It’s handy for identifying builds.
     ```bash
     git describe
     ```

### 4. **Git Tag - Lightweight and Annotated Tags**

   - **Creating Lightweight Tags**:
     ```bash
     git tag <tag-name>
     ```
   - **Creating Annotated Tags**:
     ```bash
     git tag -a <tag-name> -m "Message"
     ```
   - **Pushing Tags to Remote**:
     ```bash
     git push origin <tag-name>
     ```
   - **Push All Tags**:
     ```bash
     git push origin --tags
     ```

### 5. **Git Blame** - View Line-by-Line History

   - **Git Blame** shows who last modified each line in a file, which is useful for tracking down the origin of specific lines in code.
     ```bash
     git blame filename
     ```

### 6. **Git Apply** - Applying Patches

   - **Git Apply** applies a patch file to the working directory. This is useful for applying changes from other repositories or from contributors.
     ```bash
     git apply <patch-file>
     ```

### 7. **Git Diff with Additional Options**

   - **Check Only Names of Changed Files**:
     ```bash
     git diff --name-only
     ```
   - **View Changes by Commit**:
     ```bash
     git diff <commit-hash1> <commit-hash2>
     ```
   - **Check Word-Level Differences**:
     ```bash
     git diff --word-diff
     ```

### 8. **Git Log Options**

   - **Compact One-Line Summary**:
     ```bash
     git log --oneline
     ```
   - **Graphical History View**:
     ```bash
     git log --graph --all --decorate --oneline
     ```
   - **Search Commit Messages**:
     ```bash
     git log --grep="search-term"
     ```

### 9. **Git Show** - View Commit Details

   - **Git Show** provides detailed information about a specific commit, including the diff and metadata.
     ```bash
     git show <commit-hash>
     ```

### 10. **Git Reset with Mixed Mode**

   - **Git Reset Mixed**: Resets the staging area to match the specified commit but leaves the working directory unchanged. It’s helpful when you want to unstage changes without losing them.
     ```bash
     git reset --mixed HEAD~1
     ```

### 11. **Git Stash Advanced Options**

   - **Stash with a Message**:
     ```bash
     git stash push -m "Description of stash"
     ```
   - **Stash Only Untracked Files**:
     ```bash
     git stash --include-untracked
     ```
   - **Apply and Drop Stash in One Step**:
     ```bash
     git stash pop
     ```

### 12. **Git Prune** - Cleanup Unreachable Objects

   - **Git Prune** removes objects that are no longer referenced by any branch, which is often useful after removing branches.
     ```bash
     git prune
     ```

### 13. **Git Graft** - Connect Disconnected History

   - **Git Graft** allows you to manually connect histories of two commits when dealing with a disconnected history. Although rarely used, it can be helpful in repositories with imported histories.

### 14. **Git Archive with Specific Path**

   - **Export a Specific Directory**: Use `git archive` to create an archive file with only specific directories included.
     ```bash
     git archive --format=zip --output=output.zip HEAD path/to/directory
     ```

### 15. **Git Config for Custom Aliases**

   - **Custom Git Aliases**: Add common commands as shortcuts.
     ```bash
     git config --global alias.co checkout
     git config --global alias.br branch
     git config --global alias.ci commit
     git config --global alias.st status
     ```

### 16. **Git Submodule Advanced Commands**

   - **Update All Submodules**:
     ```bash
     git submodule update --init --recursive
     ```
   - **Remove a Submodule**:
     1. Delete the submodule from `.gitmodules` and `.git/config`.
     2. Remove the submodule files:
        ```bash
        git rm --cached path/to/submodule
        ```

### 17. **Git Bundle** - Backup or Transfer Repository

   - **Git Bundle** lets you create a single file that contains the entire Git history, useful for backups or sharing.
     ```bash
     git bundle create <file.bundle> --all
     ```
   - To clone from a bundle:
     ```bash
     git clone <file.bundle>
     ```

### 18. **Git Fast-Forward Only Merge**

   - **Fast-Forward Only**: Forces Git to only merge if it can be done with a fast-forward, which means it won’t create a merge commit.
     ```bash
     git merge --ff-only <branch-name>
     ```

### 19. **Git FSCK (File System Consistency Check)**

   - **Check Repository Consistency**: `git fsck` verifies the integrity of the repository by identifying any corrupt objects.
     ```bash
     git fsck
     ```

### 20. **Git Cherry-Pick Multiple Commits**

   - **Cherry-Pick a Range of Commits**: Selectively apply multiple commits to another branch.
     ```bash
     git cherry-pick <start-commit>^..<end-commit>
     ```

---

Each of these commands serves a unique purpose and can help in different aspects of managing and troubleshooting Git repositories. 

Git has a vast number of commands—over **160 main commands** and **dozens of options** within each command. Here’s a breakdown of the main command types and their purposes:

### Categories of Git Commands

1. **Basic Commands** - Commonly used for everyday work:
   - `git init`, `git clone`, `git add`, `git commit`, `git status`, `git push`, `git pull`, `git log`

2. **Branching and Merging** - Managing branches and combining code:
   - `git branch`, `git checkout`, `git switch`, `git merge`, `git rebase`, `git cherry-pick`, `git tag`

3. **Inspection and Comparison** - Viewing changes and differences:
   - `git diff`, `git log`, `git show`, `git blame`, `git reflog`

4. **Rewriting History** - Modifying commit history:
   - `git commit --amend`, `git rebase -i`, `git reset`, `git revert`, `git filter-branch`, `bfg`

5. **Stashing and Cleaning** - Saving or discarding changes temporarily:
   - `git stash`, `git stash pop`, `git clean`

6. **Collaboration** - Working with remote repositories:
   - `git fetch`, `git push`, `git pull`, `git remote`, `git submodule`

7. **Configuration and Setup** - Setting preferences and global settings:
   - `git config`, `git init`, `git clone`, `git remote add`

8. **Advanced Commands** - Specialized commands for complex workflows:
   - `git bisect`, `git cherry`, `git grep`, `git worktree`, `git sparse-checkout`, `git bundle`, `git archive`

9. **Utility and Debugging** - Maintenance, debugging, and recovery tools:
   - `git fsck`, `git gc`, `git prune`, `git repack`, `git replace`

10. **Hooks** - Automating actions at key stages:
    - `pre-commit`, `pre-push`, `post-merge`, and other hooks

### Complete Git Command List

Here’s a **non-exhaustive list of Git commands** to give you a better sense of Git's full command set:

- **General Commands**: `git help`, `git version`
- **Branching & Tagging**: `git branch`, `git checkout`, `git merge`, `git tag`, `git worktree`
- **Commit & History**: `git commit`, `git log`, `git reflog`, `git show`, `git blame`, `git diff`
- **Configuration**: `git config`, `git init`, `git clone`
- **Networking**: `git fetch`, `git push`, `git pull`, `git remote`
- **Patch & Revision Handling**: `git cherry`, `git format-patch`, `git apply`, `git am`
- **Archiving & Bundling**: `git archive`, `git bundle`
- **File System & Cleanup**: `git fsck`, `git gc`, `git prune`, `git repack`
- **Others**: `git describe`, `git grep`, `git bisect`, `git stash`, `git submodule`

### Git Documentation

You can find a complete list of Git commands in the [official Git documentation](https://git-scm.com/docs) or by using:

```bash
git help -a  # Shows all available git commands
```

Each command has numerous options, so the actual number of unique command usages is extensive, likely numbering in the thousands when considering all variations! Let me know if there’s a specific command you’d like to explore further.