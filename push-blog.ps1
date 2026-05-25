$env:Path = "C:\Users\DEHGHAMO\tools\PortableGit\cmd;" + $env:Path

git add .
git commit -m "$args"
git push origin main