import re
from pathlib import Path

'''
This is unnecessary now
'''

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

def convert_mysql_to_sqlite(infile, outfile):
    # Patterns for MySQL-specific syntax
    mysql_comment = re.compile(r"/\*![0-9]+.*?\*/;?")
    engine_clause = re.compile(r"ENGINE=\w+")
    auto_inc_clause = re.compile(r"AUTO_INCREMENT=\d+")
    charset_clause = re.compile(r"DEFAULT CHARSET=\w+")
    collate_clause = re.compile(r"COLLATE=\w+")
    unsigned = re.compile(r"\bunsigned\b", re.IGNORECASE)

    # Capture KEY definitions to emit later
    pending_indexes = []

    # Type conversions
    def convert_types(line):
        # Remove length specifiers from TEXT and VARCHAR
        line = re.sub(r"TEXT\(\d+\)", "TEXT", line, flags=re.IGNORECASE)
        line = re.sub(r"VARCHAR\(\d+\)", "TEXT", line, flags=re.IGNORECASE)

        # Convert MySQL integer types
        line = re.sub(r"int\(\d+\)", "INTEGER", line, flags=re.IGNORECASE)
        line = re.sub(r"tinyint\(1\)", "INTEGER", line, flags=re.IGNORECASE)

        # Convert longtext/mediumtext
        line = re.sub(r"longtext", "TEXT", line, flags=re.IGNORECASE)
        line = re.sub(r"mediumtext", "TEXT", line, flags=re.IGNORECASE)

        return line

    with open(infile, "r", encoding="utf-8", errors="ignore") as fin, \
         open(outfile, "w", encoding="utf-8") as fout:

        inside_table = False
        current_table = None

        for raw_line in fin:
            line = raw_line.strip()

            # Skip MySQL SET commands and comments
            if line.startswith("SET ") or line.startswith("--"):
                continue

            # Remove MySQL versioned comments
            line = mysql_comment.sub("", line)

            # Remove MySQL table options
            line = engine_clause.sub("", line)
            line = auto_inc_clause.sub("", line)
            line = charset_clause.sub("", line)
            line = collate_clause.sub("", line)

            # Remove UNSIGNED
            line = unsigned.sub("", line)

            # Remove backticks
            line = line.replace("`", "")

            # Detect CREATE TABLE start
            if line.upper().startswith("CREATE TABLE"):
                inside_table = True
                # Extract table name
                current_table = re.findall(r"CREATE TABLE (\w+)", line, re.IGNORECASE)[0]
                fout.write(line + "\n")
                continue

            # Detect end of CREATE TABLE
            if inside_table and line.startswith(");"):
                inside_table = False
                fout.write(");\n")

                # Emit pending indexes for this table
                for idx in pending_indexes:
                    fout.write(idx + "\n")
                pending_indexes = []
                continue

            # Inside CREATE TABLE block
            if inside_table:
                # Detect MySQL KEY definitions
                if line.upper().startswith("KEY "):
                    # Extract index name and column
                    m = re.findall(r"KEY (\w+) \((\w+)\)", line)
                    if m:
                        idx_name, col = m[0]
                        pending_indexes.append(
                            f"CREATE INDEX {idx_name} ON {current_table}({col});"
                        )
                    continue

                # Convert AUTO_INCREMENT primary key
                if "AUTO_INCREMENT" in line.upper():
                    line = re.sub(
                        r"(\w+)\s+INTEGER.*AUTO_INCREMENT.*",
                        r"\1 INTEGER PRIMARY KEY AUTOINCREMENT,",
                        line,
                        flags=re.IGNORECASE
                    )

                # Convert types
                line = convert_types(line)

                fout.write(line + "\n")
                continue

            # Outside CREATE TABLE
            if line:
                fout.write(line + "\n")

convert_mysql_to_sqlite(str(DATA_DIR / 'qmdb__v1_3__102019.sql'), str(DATA_DIR / 'qmdb__v1_3__converted.sql'))