# 0) Stop tracking & ignore Terraform junk going forward
printf '\n# Terraform\n.terraform/\n*.tfstate\n*.tfstate.*\n.terraform.lock.hcl\n' >> .gitignore
git add .gitignore && git commit -m "chore: ignore Terraform state/cache" || true
git rm -r --cached -- terraform/.terraform terraform/terraform.tfstate terraform/terraform.tfstate.backup terraform/.terraform.lock.hcl 2>/dev/null || true
git commit -m "chore: untrack Terraform cache/state" || true

# 1) Remove the big files from history (use filter-repo; falls back to python module if needed)
( command -v git-filter-repo >/dev/null \
  && git filter-repo --invert-paths --path-glob 'terraform/.terraform/**' --path 'terraform/terraform.tfstate' --path 'terraform/terraform.tfstate.backup' --path 'terraform/.terraform.lock.hcl' --strip-blobs-bigger-than 100M --force ) \
|| ( python3 -m pip install --user git-filter-repo && python3 -m git_filter_repo --invert-paths --path-glob 'terraform/.terraform/**' --path 'terraform/terraform.tfstate' --path 'terraform/terraform.tfstate.backup' --path 'terraform/.terraform.lock.hcl' --strip-blobs-bigger-than 100M --force )

# 2) Nuke local cache to avoid re-adding, then force-push rewritten history
rm -rf terraform/.terraform
git push origin --force --all
git push origin --force --tags

# 3) Re-init Terraform locally (since .terraform/ is gone)
( cd terraform && terraform init -reconfigure )

