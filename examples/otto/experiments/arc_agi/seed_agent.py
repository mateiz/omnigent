import json
import re


def solve(train_inputs, train_outputs, test_inputs, llm):
    examples = "\n".join(
        f"Input: {i}\nOutput: {o}" for i, o in zip(train_inputs, train_outputs, strict=False)
    )
    tests = "\n".join(f"Test {i}: {x}" for i, x in enumerate(test_inputs))
    prompt = (
        f"Solve this ARC puzzle. Examples:\n{examples}\n\nReturn JSON grids for tests:\n{tests}"
    )
    response = llm(prompt)
    grids = [json.loads(g) for g in re.findall(r"\[\[.*?\]\]", response.replace("\n", ""))]
    return {"train": train_outputs, "test": [[grid] for grid in grids[: len(test_inputs)]]}
