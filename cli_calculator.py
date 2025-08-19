#!/usr/bin/env python3
"""
CLI Calculator (safe eval, variables, math functions)

Usage:
  python cli_calculator.py
Examples:
  > 2 + 3*4
  14
  > x = 5
  x = 5
  > sqrt(x**2 + 11)
  12.083045973594572
  > ans / 2
  6.041522986797286
Commands:
  :help  :vars  :clear  :quit
"""

import ast
import math
import operator
import sys

# Allowed operators
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions (name -> callable)
FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": lambda x, base=10: math.log(x, base),
    "ln": math.log,       # natural log
    "exp": math.exp,
    "abs": abs,
    "round": round,
}

# Allowed constants
CONSTS = {"pi": math.pi, "e": math.e}

class SafeEvaluator(ast.NodeVisitor):
    """
    Safely evaluate arithmetic expressions with a restricted AST.
    Supports numbers, variables, the operators in BIN_OPS/UNARY_OPS,
    function calls from FUNCS, and parentheses.
    """

    def __init__(self, env):
        self.env = env  # variable environment (dict)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Module(self, node):  # for completeness when using ast.parse(..., mode="exec")
        if len(node.body) != 1:
            raise ValueError("Only single expressions/assignments allowed")
        return self.visit(node.body[0])

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise ValueError("Only simple assignments supported (e.g., x = 3)")
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise ValueError("Left side must be a variable name")
        value = self.visit(node.value)
        self.env[target.id] = value
        return (target.id, value)

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
        if node.id in CONSTS:
            return CONSTS[node.id]
        if node.id in FUNCS:
            return FUNCS[node.id]
        raise NameError(f"Unknown name: {node.id}")

    def visit_Call(self, node):
        func = self.visit(node.func)
        if func not in FUNCS.values():
            raise ValueError("Only approved functions are allowed")
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in BIN_OPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        # Guard: disallow 0 division in a friendly way
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        return UNARY_OPS[op_type](operand)

    def visit_IfExp(self, node):
        raise ValueError("Conditional expressions are not allowed")

    def visit_Compare(self, node):
        raise ValueError("Comparisons are not allowed")

    def visit_BoolOp(self, node):
        raise ValueError("Boolean operators are not allowed")

    def visit_Lambda(self, node):
        raise ValueError("Lambdas are not allowed")

    def visit_Attribute(self, node):
        raise ValueError("Attributes are not allowed")

    def visit_Subscript(self, node):
        raise ValueError("Indexing/slicing is not allowed")

    def visit_List(self, node):
        raise ValueError("Lists are not allowed")

    def visit_Tuple(self, node):
        raise ValueError("Tuples are not allowed")

    def visit_Dict(self, node):
        raise ValueError("Dicts are not allowed")

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric literals are allowed")

    # Python <3.8 compatibility:
    def visit_Num(self, node):  # pragma: no cover
        return node.n

def eval_line(s, env):
    """
    Evaluate a single line which can be an assignment (x=...) or an expression.
    Returns either the numeric result, or a tuple ('var', value) for assignments.
    """
    s = s.strip()
    # Try assignment (exec) first if it looks like one; else expression
    try:
        if "=" in s and not s.startswith(("==", "!=")):
            tree = ast.parse(s, mode="exec")
            return SafeEvaluator(env).visit(tree)
        else:
            tree = ast.parse(s, mode="eval")
            return SafeEvaluator(env).visit(tree)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e.msg}")

def repl():
    print("CLI Calculator — type :help for commands. Ctrl+C or :quit to exit.")
    env = {"ans": 0.0}
    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            if line.startswith(":"):
                cmd = line.lower().strip()
                if cmd in (":quit", ":q", ":exit"):
                    print("Goodbye!")
                    return
                if cmd in (":help", ":h"):
                    print(
                        "Commands: :help  :vars  :clear  :quit\n"
                        "Features: numbers, + - * / // % **, parentheses, variables, functions\n"
                        "Functions: sqrt, sin, cos, tan, log(x, base=10), ln, exp, abs, round\n"
                        "Constants: pi, e | Last result: ans\n"
                        "Examples:\n"
                        "  2 + 3*4\n"
                        "  x = 5\n"
                        "  sqrt(x**2 + 11)\n"
                        "  log(1000, base=10)\n"
                        "  ans / 2\n"
                    )
                    continue
                if cmd in (":vars", ":v"):
                    keys = sorted(k for k in env.keys() if k not in ("ans",))
                    if not keys:
                        print("(no variables set)")
                    else:
                        for k in keys:
                            print(f"{k} = {env[k]}")
                    print(f"ans = {env['ans']}")
                    continue
                if cmd in (":clear", ":c"):
                    env = {"ans": 0.0}
                    print("Cleared variables. ans = 0.0")
                    continue
                print("Unknown command. Try :help")
                continue

            result = eval_line(line, env)
            if isinstance(result, tuple):  # assignment
                var, val = result
                # Store also in ans for convenience
                env["ans"] = float(val)
                print(f"{var} = {val}")
            else:
                env["ans"] = float(result)
                print(result)
        except ZeroDivisionError as e:
            print(f"Error: {e}")
        except NameError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except EOFError:
            print("\nGoodbye!")
            sys.exit(0)

if __name__ == "__main__":
    repl()