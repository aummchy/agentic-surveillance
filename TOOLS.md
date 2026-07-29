# TOOLS.md — Developer Tools & Techniques

## AI Agent Tools

| Tool | Purpose | Install |
|------|---------|---------|
| opencode | AI coding assistant | Pre-installed |
| AGENTS.md | AI agent context | Read automatically by agents |
| repomix | Codebase digest for LLMs | npx repomix |
| ctags | Symbol index | winget install universal-ctags |
| ast-grep | Semantic code search | npm install -g @ast-grep/cli |

## Repomix Commands

```bash
npx repomix                    # Generate repomix-output.xml
npx repomix --style json       # JSON format
```

## ctags Commands

```bash
ctags -R --languages=Python --exclude=venv --exclude=__pycache__ -f tags .
```

## ast-grep Commands

```bash
sg --pattern 'def $FUNC($$$): $$$' --lang python    # Find all function defs
sg --pattern 'set_embedding($$$)' --lang python       # Find all calls
sg -p 'if $COND:' --lang python --json               # JSON output
```

## Useful Commands

```bash
python main.py                 # Run surveillance pipeline
python -m pytest tests/ -v     # Run tests
cd dashboard/frontend && npm run dev  # Run dashboard
```
