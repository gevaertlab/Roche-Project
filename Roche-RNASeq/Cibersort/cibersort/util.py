def format_graph(ax, title, x_label, y_label):
    ax.set_title(title, fontsize = 18,weight = "bold")
    ax.set_xlabel(x_label, fontsize = 16, weight = "bold")
    ax.set_ylabel(y_label, fontsize = 16, weight = "bold")
    ax.tick_params(labelsize = 18)


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def construct_pairs(cat1, cat2):
    res = []
    for i in cat1:   
        res = res + [((i, clar),(i, clar)) for clar in cat2]
    return res


