def is_selected_from_n_clicks(n_clicks):
    if n_clicks is None:
        return False
    if n_clicks % 2 == 0:
        return False
    if n_clicks % 2 == 1:
        return True