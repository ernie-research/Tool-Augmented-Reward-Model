def calculator(expr):
    """This tool allows you to use the numexpr library to evaluate expressions,
    
    Example:
        - 2 + 2
        - 2 * 2 
    """
    import numexpr 
    try:
        return numexpr.evaluate(expr)
    except Exception as e:
        return f"Error: {e}, try again and only use a numerical expression"