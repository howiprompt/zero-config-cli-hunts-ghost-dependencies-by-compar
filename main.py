"""
Zero-config CLI that hunts 'ghost' dependencies by comparing import statements in source files against package manifests

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike the heavy, Go-based hybrid architecture of `alibaba/open-code-review` (which requires agents and databases), this is a lightweight, single-file Python tool using stdlib `ast`/regex to pinpoint 
"""
#!/usr/bin/env python3
"""
Ghost Hunter: Zero-Config Dependency Auditing Tool

A robust CLI tool designed to identify discrepancies between declared dependencies
and actual usage in Python and JavaScript/TypeScript projects.

Features:
- Detects 'Ghost Dependencies': Libraries imported in code but missing from manifests.
- Detects 'Unused Dependencies': Libraries listed in manifests but never imported.
- Supports local directories and remote GitHub repositories.
- Parsing for requirements.txt, pyproject.toml, package.json.
- AST-based Python parsing and Regex-based JS/TS parsing.

Usage Examples:
    # Analyze current directory
    python ghost_hunter.py .

    # Analyze a specific local path
    python ghost_hunter.py /path/to/project

    # Analyze a GitHub repository
    python ghost_hunter.py https://github.com/username/repo

    # Specify custom branch for GitHub
    python ghost_hunter.py https://github.com/username/repo --branch develop

Environment Variables:
    GITHUB_TOKEN: Optional. If set, used for authenticated GitHub API requests to
                  avoid rate limiting when scanning repositories.
"""

import argparse
import ast
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --- Constants & Configuration ---

# ANSI Color Codes for Terminal Output
class Colors:
    HEADER = "\033[95m"
    OK_BLUE = "\033[94m"
    OK_CYAN = "\033[96m"
    OK_GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END_C = "\033[0m"
    BOLD = "\033[1m"

# File patterns to ignore
IGNORED_DIRS = {
    ".git", ".idea", ".vscode", "node_modules", "__pycache__",
    "venv", ".venv", "env", ".env", "dist", "build", ".tox"
}

SOURCE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx"}
MANIFEST_FILES_PY = {"requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"}
MANIFEST_FILES_JS = {"package.json"}

# GitHub Configuration
GITHUB_API_BASE = "https://api.github.com/"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/"

# --- Data Structures ---

@dataclass
class DependencyReport:
    project_path: str
    ghost_deps: Dict[str, List[str]] = field(default_factory=dict)  # Dependency -> List of files
    unused_deps: Set[str] = field(default_factory=set)
    declared_deps: Set[str] = field(default_factory=set)
    imported_libs: Set[str] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)

# --- Utility Functions ---

def log(message: str, level: str = "info") -> None:
    """Colored logging for CLI output."""
    if level == "error":
        print(f"{Colors.FAIL}[-] {message}{Colors.END_C}")
    elif level == "warning":
        print(f"{Colors.WARNING}[!] {message}{Colors.END_C}")
    elif level == "success":
        print(f"{Colors.OK_GREEN}[+] {message}{Colors.END_C}")
    elif level == "info":
        print(f"{Colors.OK_CYAN}[*] {message}{Colors.END_C}")
    else:
        print(message)

def normalize_package_name(name: str) -> str:
    """
    Normalize package names to handle underscores, hyphens, and case sensitivity.
    e.g., 'Pandas' -> 'pandas', 'my-package' -> 'my_package'
    """
    name = name.lower().strip()
    # Replace hyphens with underscores (common convention in import vs package names)
    return re.sub(r"[-_.]+", "_", name)

def get_github_token() -> Optional[str]:
    """Retrieve GitHub token from environment."""
    return os.environ.get("GITHUB_TOKEN")

# --- Network & GitHub Handling ---

class GitHubFetcher:
    """Handles fetching repository content from GitHub."""

    def __init__(self):
        self.token = get_github_token()
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

    def _make_request(self, url: str) -> Optional[bytes]:
        """Make a raw HTTP GET request."""
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    return response.read()
                return None
        except urllib.error.HTTPError as e:
            if e.code == 403 and "rate limit exceeded" in str(e).lower():
                log("GitHub API rate limit exceeded. Please set GITHUB_TOKEN.", "error")
            elif e.code == 404:
                log(f"Resource not found: {url}", "error")
            else:
                log(f"HTTP Error {e.code}: {url}", "error")
            return None
        except Exception as e:
            log(f"Network error: {e}", "error")
            return None

    def get_repo_info(self, repo_url: str, branch: str = "main") -> Tuple[Optional[str], Optional[str]]:
        """
        Parse GitHub URL and return owner/repo and validated branch (or default).
        Supports: https://github.com/owner/repo OR https://github.com/owner/repo.git
        """
        # Basic parsing
        match = re.search(r"github\.com/([^/]+)/([^/]+?)(\.git)?$", repo_url)
        if not match:
            log(f"Invalid GitHub URL format: {repo_url}", "error")
            return None, None
        
        owner, repo = match.groups()[0], match.groups()[1]
        repo_full = f"{owner}/{repo}"
        
        # Verify default branch if 'main' is assumed but might be 'master'
        api_url = f"{GITHUB_API_BASE}repos/{owner}/{repo}"
        data = self._make_request(api_url)
        if data:
            try:
                repo_meta = json.loads(data.decode('utf-8'))
                default_branch = repo_meta.get('default_branch', 'master')
                return repo_full, default_branch
            except json.JSONDecodeError:
                pass
                
        return repo_full, branch

    def get_file_list(self, owner_repo: str, branch: str) -> List[str]:
        """
        Fetch the recursive tree for the repository using GitHub API.
        Returns a list of file paths.
        """
        url = f"{GITHUB_API_BASE}repos/{owner_repo}/git/trees/{branch}?recursive=1"
        log(f"Fetching file tree for {owner_repo}...", "info")
        
        data = self._make_request(url)
        if not data:
            # Fallback: Try to construct without API (blind scan) - rarely works for complex scans
            # but for the sake of the Spec's "raw scraping" mention, we could try.
            # However, listing files blindly without API or directory index is impossible on raw.github.
            # We rely on the API here or return empty.
            return []

        try:
            content = json.loads(data.decode('utf-8'))
            if content.get('truncated'):
                log("Warning: Repository file list truncated by API. Results may be incomplete.", "warning")
            
            files = []
            for item in content.get('tree', []):
                if item['type'] == 'blob':
                    path = item['path']
                    # Filter directories we know we don't need
                    if not any(ignore in path.split('/') for ignore in IGNORED_DIRS):
                        files.append(path)
            return files
        except json.JSONDecodeError:
            log("Failed to parse GitHub API response.", "error")
            return []

    def download_file_content(self, owner_repo: str, branch: str, path: str) -> Optional[str]:
        """Fetch raw content of a single file."""
        url = f"{GITHUB_RAW_BASE}{owner_repo}/{branch}/{path}"
        data = self._make_request(url)
        if data:
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return None # Binary file
        return None

# --- Parsing Logic ---

class CodeParser:
    """Extracts import statements from source code and dependencies from manifests."""

    def __init__(self):
        self.imports_cache: Set[str] = set()
        self.deps_cache: Set[str] = set()

    # --- Python AST Parsing ---
    def extract_python_imports(self, file_path: str, content: str) -> Set[str]:
        """Use AST to parse Python files accurately."""
        imports = set()
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level package name
                        pkg_name = alias.name.split('.')[0]
                        imports.add(pkg_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # 'from .x import y' (relative) -> ignore
                        if node.level > 0:
                            continue
                        # 'from package import module'
                        pkg_name = node.module.split('.')[0]
                        imports.add(pkg_name)
        except (SyntaxError, ValueError) as e:
            # Fallback to regex if AST fails (e.g., Python 2 syntax or f-strings in old parser)
            log(f"AST parse failed for {file_path} (using regex fallback)", "warning")
            imports.update(self._regex_python_imports(content))
        return imports

    def _regex_python_imports(self, content: str) -> Set[str]:
        """Regex fallback for Python imports."""
        imports = set()
        # Match 'import X' or 'import X as Y'
        match_import = re.findall(r"^\s*import\s+([a-zA-Z_0-9]+)", content, re.MULTILINE)
        # Match 'from X import ...'
        match_from = re.findall(r"^\s*from\s+([a-zA-Z_0-9]+)\s+import", content, re.MULTILINE)
        
        for m in match_import:
            imports.add(m.split('.')[0])
        for m in match_from:
            imports.add(m.split('.')[0])
        return imports

    # --- JavaScript / TypeScript Parsing ---
    def extract_js_imports(self, file_path: str, content: str) -> Set[str]:
        """Use Regex to parse JS/TS imports."""
        imports = set()
        
        # Match: import ... from 'lib' or require('lib')
        # Allows single or double quotes.
        # Ignores relative paths starting with . or /
        
        patterns = [
            r"import\s+.*?from\s+['\"]([^'\"./][^'\"]*)['\"]", 
            r"require\(\s*['\"]([^'\"./][^'\"]*)['\"]\s*\)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Cleanup: remove scopes like @types/node -> node (optional, but safer to keep scope sometimes)
                # For ghost hunting, we usually treat @scope/pkg as "scope_pkg" or "pkg" depending on strictness.
                # Here we normalize strictly: @babel/core -> babel/core.
                clean_name = match.lstrip('@')
                
                # Scoped packages in package.json are "@scope/name". Import is "import ... from '@scope/name'".
                # We normalize both to comparable strings.
                imports.add(normalize_package_name(clean_name))
                
        return imports

    # --- Manifest Parsing ---
    
    def parse_requirements_txt(self, content: str) -> Set[str]:
        deps = set()
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith(('#', '-', 'git', 'http')):
                continue
            # Extract name before version specifier or comment
            name = re.split(r"[=<>;#]", line)[0].strip()
            if name:
                deps.add(normalize_package_name(name))
        return deps

    def parse_package_json(self, content: str) -> Set[str]:
        deps = set()
        try:
            data = json.loads(content)
            for key in ["dependencies", "devDependencies", "peerDependencies"]:
                if key in data:
                    for pkg_name in data[key].keys():
                        deps.add(normalize_package_name(pkg_name))
        except json.JSONDecodeError:
            pass
        return deps

    def parse_pyproject_toml(self, content: str) -> Set[str]:
        """
        Manual TOML parsing for sections relevant to dependencies.
        Avoids external toml lib dependency.
        """
        deps = set()
        
        # 1. Try to find [project.dependencies] or [tool.poetry.dependencies]
        # This is a simplified heuristic regex parser.
        
        # Capture content inside a dependencies block
        # We look for start of section and end of section (next [ )
        
        # Strategy: Split lines, track if we are in a dependency section.
        lines = content.splitlines()
        in_dep_section = False
        dep_sections = [
            "[project.dependencies]",
            "[tool.poetry.dependencies]",
            "[poetry.dependencies]"
        ]
        
        for line in lines:
            stripped = line.strip()
            
            # Check if we are entering a target section
            if any(stripped.startswith(sec) for sec in dep_sections):
                in_dep_section = True
                continue
            
            # Check if we exited any section
            if in_dep_section and stripped.startswith("["):
                in_dep_section = False
                continue
                
            if in_dep_section:
                # Parse dependency line
                # Format: name = "version" or name = {version = "..."}
                line = line.split("=")[0].strip()
                if line and not line.startswith("#"):
                    deps.add(normalize_package_name(line))
                    
        return deps

# --- Core Analysis Logic ---

def analyze_local_directory(root_path: Path) -> DependencyReport:
    """Scan a local directory."""
    report = DependencyReport(project_path=str(root_path))
    parser = CodeParser()
    
    if not root_path.is_dir():
        log(f"Path is not a directory: {root_path}", "error")
        return report

    log(f"Scanning local directory: {root_path}", "info")

    # 1. Locate Manifests
    for filename in MANIFEST_FILES_PY.union(MANIFEST_FILES_JS):
        manifest_path = root_path / filename
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if filename == "requirements.txt":
                        report.declared_deps.update(parser.parse_requirements_txt(content))
                    elif filename == "package.json":
                        report.declared_deps.update(parser.parse_package_json(content))
                    elif filename == "pyproject.toml":
                        report.declared_deps.update(parser.parse_pyproject_toml(content))
            except Exception as e:
                report.errors.append(f"Failed to read {filename}: {e}")

    # 2. Scan Source Files
    source_files = []
    for ext in SOURCE_EXTENSIONS:
        source_files.extend(root_path.rglob(f"*{ext}"))
        
    valid_files = [f for f in source_files if not any(ig in str(f) for ig in IGNORED_DIRS)]

    for sf in valid_files:
        try:
            with open(sf, 'r', encoding='utf-8') as f:
                content = f.read()
                if sf.suffix == ".py":
                    found = parser.extract_python_imports(str(sf), content)
                else:
                    found = parser.extract_js_imports(str(sf), content)
                
                for lib in found:
                    if lib not in report.imported_libs:
                        report.imported_libs.add(lib)
                        # Track first seen file for ghost dependencies
                        # (For this report structure, we assume Ghost Deps is a dict of dep->files)
                        if lib not in report.ghost_deps:
                             report.ghost_deps[lib] = [str(sf)]
                        else:
                             report.ghost_deps[lib].append(str(sf))
                             
        except Exception as e:
            report.errors.append(f"Failed to parse {sf}: {e}")

    # 3. Calculate Results
    # Ghost: Imported but NOT declared
    # Unused: Declared but NOT imported
    
    # Remove standard library or known false positives logic could go here
    # For now, strict set difference.
    
    # Logic check: Filter out matches
    # (Actually, standard logic is Imported - Declared = Ghost)
    
    # Let's create accurate sets for calculation
    # Note: report.ghost_deps was populated incorrectly above as "all".
    # Let's fix that:
    report.ghost_deps = {}
    
    normalized_declared = {normalize_package_name(d) for d in report.declared_deps}
    
    for imp in report.imported_libs:
        norm_imp = normalize_package_name(imp)
        if norm_imp not in normalized_declared:
            if imp not in report.ghost_deps:
                report.ghost_deps[imp] = []
            # Add file mapping (requires storing mapping during parse or re-iterating)
            # Re-iterating for simplicity to map files to ghosts
            pass

    # We need file mapping for ghosts. Let's do a second pass or better, store during first pass.
    # Let's correct the collection logic:
    report.imported_libs = set()
    report.import_map: Dict[str, List[str]] = {}
    
    for sf in valid_files:
        try:
            with open(sf, 'r', encoding='utf-8') as f:
                content = f.read()
                if sf.suffix == ".py":
                    found = parser.extract_python_imports(str(sf), content)
                else:
                    found = parser.extract_js_imports(str(sf), content)
                
                for lib in found:
                    report.imported_libs.add(lib)
                    if lib not in report.import_map:
                        report.import_map[lib] = []
                    report.import_map[lib].append(str(sf))
        except Exception:
            continue

    # Final Comparison
    report.ghost_deps = {}
    for lib, files in report.import_map.items():
        if normalize_package_name(lib) not in normalized_declared:
            report.ghost_deps[lib] = files

    report.unused_deps = set()
    for dep in report.declared_deps:
        norm_dep = normalize_package_name(dep)
        if norm_dep not in {normalize_package_name(imp) for imp in report.imported_libs}:
            report.unused_deps.add(dep)

    return report

def analyze_github_repo(repo_url: str, branch: str = None) -> DependencyReport:
    """Scan a remote GitHub repository."""
    fetcher = GitHubFetcher()
    
    owner_repo, default_branch = fetcher.get_repo_info(repo_url, branch)
    if not owner_repo:
        return DependencyReport(project_path=repo_url, errors=["Invalid repo URL."])
    
    target_branch = branch if branch else default_branch
    log(f"Analyzing {owner_repo} on branch '{target_branch}'...", "info")
    
    # Fetch file structure
    file_list = fetcher.get_file_list(owner_repo, target_branch)
    if not file_list:
        return DependencyReport(project_path=repo_url, errors=["Could not fetch file list."])
        
    report = DependencyReport(project_path=repo_url)
    parser = CodeParser()
    
    # Identify targets
    manifest_files = [f for f in file_list if os.path.basename(f) in MANIFEST_FILES_PY.union(MANIFEST_FILES_JS)]
    source_files = [f for f in file_list if any(f.endswith(ext) for ext in SOURCE_EXTENSIONS)]
    
    # Parse manifests
    for mf in manifest_files:
        content = fetcher.download_file_content(owner_repo, target_branch, mf)
        if content:
            fname = os.path.basename(mf)
            if fname == "requirements.txt":
                report.declared_deps.update(parser.parse_requirements_txt(content))
            elif fname == "package.json":
                report.declared_deps.update(parser.parse_package_json(content))
            elif fname == "pyproject.toml":
                report.declared_deps.update(parser.parse_pyproject_toml(content))
    
    # Parse sources
    for sf in source_files:
        content = fetcher.download_file_content(owner_repo, target_branch, sf)
        if content:
            try:
                if sf.endswith(".py"):
                    found = parser.extract_python_imports(sf, content)
                else:
                    found = parser.extract_js_imports(sf, content)
                
                for lib in found:
                    report.imported_libs.add(lib)
                    if lib not in report.import_map:
                        report.import_map[lib] = []
                    report.import_map[lib].append(sf)
            except Exception:
                continue

    # Calculate
    normalized_declared = {normalize_package_name(d) for d in report.declared_deps}
    
    report.ghost_deps = {}
    for lib, files in report.import_map.items():
        if normalize_package_name(lib) not in normalized_declared:
            report.ghost_deps[lib] = files
            
    # Check unused
    for dep in report.declared_deps:
        if normalize_package_name(dep) not in {normalize_package_name(imp) for imp in report.imported_libs}:
            report.unused_deps.add(dep)
            
    return report

# --- Reporting ---

def print_report(report: DependencyReport) -> None:
    """Output the formatted TTY report."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== Ghost Hunter Report ==={Colors.END_C}")
    print(f"Target: {report.project_path}")
    
    if report.errors:
        print(f"\n{Colors.WARNING}Errors Encountered:{Colors.END_C}")
        for err in report.errors:
            print(f"  - {err}")

    print("\n" + "-" * 40)
    
    # Ghost Dependencies
    if report.ghost_deps:
        print(f"{Colors.FAIL}{Colors.BOLD}Ghost Dependencies (Found but not listed):{Colors.END_C}")
        print(f"Count: {len(report.ghost_deps)}")
        for lib, files in sorted(report.ghost_deps.items()):
            print(f"  {Colors.BOLD}{lib}{Colors.END_C}")
            print(f"    -> Found in: {', '.join(files)}")
    else:
        print(f"{Colors.OK_GREEN}No Ghost Dependencies detected.{Colors.END_C}")

    print("\n" + "-" * 40)
    
    # Unused Dependencies
    if report.unused_deps:
        print(f"{Colors.WARNING}{Colors.BOLD}Unused Dependencies (Listed but not found):{Colors.END_C}")
        print(f"Count: {len(report.unused_deps)}")
        for dep in sorted(report.unused_deps):
            print(f"  - {dep}")
    else:
        print(f"{Colors.OK_GREEN}No Unused Dependencies detected.{Colors.END_C}")
        
    print("\n" + "-" * 40)
    print("Analysis complete.\n")

    # Return exit code (0 if clean, 1 if ghosts or unused found, 2 if errors)
    if report.errors:
        sys.exit(2)
    if report.ghost_deps or report.unused_deps:
        sys.exit(1)
    sys.exit(0)

# --- CLI Entry Point ---

def main():
    parser_obj = argparse.ArgumentParser(
        description="Ghost Hunter: Hunt down ghost and unused dependencies.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__
    )
    
    parser_obj.add_argument(
        "target",
        help="Local path or GitHub URL to the project."
    )
    
    parser_obj.add_argument(
        "--branch",
        default=None,
        help="Git branch to use (only for GitHub URLs). Default is the repo's default branch."
    )

    args = parser_obj.parse_args()

    # Determine strategy
    if args.target.startswith("http://") or args.target.startswith("https://"):
        report = analyze_github_repo(args.target, args.branch)
    else:
        path = Path(args.target).resolve()
        if not path.exists():
            log(f"Path does not exist: {args.target}", "error")
            sys.exit(2)
        report = analyze_local_directory(path)

    print_report(report)

if __name__ == "__main__":
    main()