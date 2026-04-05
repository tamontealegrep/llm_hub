from gitingest import ingest

summary, tree, content = ingest(".")

FILE_NAME = "code_prompt.txt"
with open(FILE_NAME, "w", encoding="utf-8") as f:
    f.write(tree)
    f.write("\n\n")
    f.write(content)

print(f"Done! File '{FILE_NAME}' generated")