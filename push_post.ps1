cd "C:\Users\DEHGHAMO\modeha.github.io"

$message = $args[0]

if ([string]::IsNullOrWhiteSpace($message)) {
    $message = "Update website"
}

git status
git add .
git commit -m "$message"
git push origin main
