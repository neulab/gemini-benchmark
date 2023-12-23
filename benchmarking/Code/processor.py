"""Perform post-processing on codex predictions."""

from typing import List

STOP_TOKEN = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n\n\n"]


def clean_prediction(text: str) -> str:  # deprecated
    """Clean codex prediction, keep the first section only."""
    lines = text.split("\n")
    lines = [lin.lstrip("#").strip() for lin in lines]
    if "" in lines:  # keep only the first section
        index = lines.index("")
        lines = lines[:index]
    lines = [l.strip() for l in lines]
    return "\n".join(lines)


class CodeProcessor:
    """Post-processor of generated code snippets."""

    def __init__(self, verbose: bool = False):
        self.stop_tokens = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n\n\n"]
        self.verbose = verbose

    def check_case_validation(self, content: str) -> bool:
        """Check if the test case is correct in syntax."""
        if not content.strip():
            return False

        try:
            indented_content = content.replace("\n", "\n    ")
            assertion_block = f"try:\n    {indented_content}\nexcept:\n    pass"
            if self.verbose:
                print(f"[assertion block]\n{assertion_block}")
            compile(assertion_block, "", "exec")
            return True
        except:
            return False

    def extract_after_first_def_newline(self, content: str) -> str:
        import re

        pattern = r"def .*?\n"
        match = re.search(pattern, content)

        if match:
            start_index = match.end()
            return content[start_index:].strip()

        return content

    def remove_code_block(self, content: str) -> str:
        return content.replace("```python", "").replace("```", "").strip()

    def truncate(self, content: str) -> str:
        for identifier in self.stop_tokens:
            if identifier in content:
                content = content.split(identifier)[0]
        return content

    def code_extract(self, content: str) -> str:
        """Extract generated code solution."""
        return self.truncate(
            self.extract_after_first_def_newline(self.remove_code_block(content))
        )


class TestProcessor:
    """Post-processor of unit test cases from model predictions."""

    def __init__(self, verify_extraction: bool = True):
        self.verify_extraction = verify_extraction
        self.stop_tokens = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

    def truncate_case(self, content: str) -> str:
        """Truncate predicted content by pre-defined stop words."""
        for identifier in self.stop_tokens:
            if identifier in content:
                content = content.split(identifier)[0]
        return content.strip()

    def dedup_assertion(self, content: str) -> str:
        """Remove deuplicate assertion lines in a single test case.
        Note that some lines are not assertions, they are not deduplicated for now.
        """
        unique_lines = []
        for line in content.split("\n"):
            if not line.strip():
                continue
            if line.startswith("assert ") and (line.strip() in unique_lines):
                continue
            unique_lines.append(line.strip())
        return "\n".join(unique_lines)

    def remove_comment(self, content: str) -> str:
        lines = []
        for line in content.split("\n"):
            if line.lstrip().startswith("# "):
                continue
            lines.append(line)
        return "\n".join(lines)

    def check_case_validation(self, content: str) -> bool:
        """Check if the test case is correct in syntax."""
        if not content.strip():
            return False
        if "assert " not in content:
            return False

        try:
            indented_content = content.replace("\n", "\n    ")
            assertion_block = f"try:\n    {indented_content}\nexcept:\n    pass"
            compile(assertion_block, "", "exec")
            return True
        except:
            return False

    def test_case_extract(
        self,
        content: str,
        entry_point: str,
        case_delimiter: str = "\n\n",
        test_entry_point: str = "candidate",
    ) -> List[str]:
        """Extract unit test cases from the bulk of prediction.
        Test cases are seperated with '\n\n' by default.
        """
        test_cases = []
        for chunk in content.split(case_delimiter):
            if not chunk.strip():
                continue
            if entry_point in chunk:
                chunk = chunk.replace(entry_point, test_entry_point)
            chunk = self.dedup_assertion(self.remove_comment(self.truncate_case(chunk)))

            if self.verify_extraction == True:
                if self.check_case_validation(chunk):
                    test_cases.append(chunk)
            else:
                test_cases.append(chunk)

        return test_cases
