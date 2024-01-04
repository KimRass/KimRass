from openpyxl.styles import PatternFill, Font
from openpyxl.styles.colors import Color


class ExcelStyler:
    def add_color_to_cell(self, cell, color="00FF4942") -> None:
        color = Color(rgb=color)
        pattern_fill = PatternFill(patternType="solid", fgColor=color)
        cell.fill = pattern_fill

    def add_color_to_font(self, cell, color="000000FF") -> None:
        cell.font = Font(color=color)

    def apply_bold_to_font(self, cell) -> None:
        cell.font = Font(bold=True)

    def apply_double_underline_to_font(self, cell) -> None:
        cell.font = Font(underline="double")

    def __ini__(self, logger=None) -> None:
        self.logger = logger
