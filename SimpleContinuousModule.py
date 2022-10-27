import mesa


class SimpleCanvas(mesa.visualization.VisualizationElement):
    local_includes = ["simple_continuous_canvas.js"]
    portrayal_method = None
    canvas_height = 500
    canvas_width = 500

    def __init__(self, portrayal_method, canvas_height=500, canvas_width=500):
        """
        Instantiate a new SimpleCanvas
        """
        super().__init__()
        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = "new Simple_Continuous_Module({}, {})".format(
            self.canvas_width, self.canvas_height
        )
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        space_state = []
        for obj in model.schedule.agents:
            portrayal = self.portrayal_method(obj)
            x, y = obj.pos
            x = (x - model.grid.x_min) / (model.grid.x_max - model.grid.x_min)
            y = (y - model.grid.y_min) / (model.grid.y_max - model.grid.y_min)
            portrayal["x"] = x
            portrayal["y"] = y
            space_state.append(portrayal)
        return space_state