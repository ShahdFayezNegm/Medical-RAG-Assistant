import re


class DocumentCleaner:
    """
    Cleans PDF pages before chunking.
    """

    def should_skip(self, doc):
        text = doc.page_content.strip()
        page = doc.metadata.get("page", -1)

        # Skip empty / very short pages
        if len(text) < 100:
            return True

        # Skip cover page
        if page == 0:
            return True

        lower_text = text.lower()

        # Skip table of contents pages
        if "contents" in lower_text:
            return True

        # Detect TOC pages with dotted leaders
        dotted_lines = len(
            re.findall(
                r"\.{5,}",
                text
            )
        )

        if dotted_lines >= 5:
            return True

        return False

    def clean_text(self, text):
        """
        Normalize whitespace.
        """

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()