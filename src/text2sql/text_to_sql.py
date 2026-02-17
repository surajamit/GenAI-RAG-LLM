def generate_sql(question):
    # simplified template
    if "count" in question.lower():
        return "SELECT COUNT(*) FROM table;"
    return "SELECT * FROM table LIMIT 10;"
