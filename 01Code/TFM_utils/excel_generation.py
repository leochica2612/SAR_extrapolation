
import pandas as pd
import numpy as np
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from TFM_utils.data_retrieve import Data_Retrieved


def _clean_value(value):
    """Convert missing or non-finite values to Excel-compatible blanks."""
    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None

    return value


def export_results_to_excel(
    all_data: Data_Retrieved,
    output_path="TFM_results.xlsx",
    stability_by_file=None,
    avoid_for_metrics = None
):
    """
    Export retrieved wind and atmospheric data to Excel.

    Parameters
    ----------
    all_data : Data_Retrieved
        Container holding the results for each processed file.
    output_path : str
        Destination Excel filename.
    stability_by_file : dict, optional
        Mapping from filename to stability label, e.g.:
        {"file_1": "vvu", "file_2": "u"}.
    """

    stability_by_file = stability_by_file or {}

    map_rows = []
    wind_rows = []
    atmospheric_rows = []
    relerr_rows = []
    abserr_rows = []
    errors_by_level = {}
    metrics_rows = []
    stadistics_wind_rows = []

    for filename, values in all_data.results.items():

        # --------------------------------------------------
        # Sheet 1: map
        # --------------------------------------------------
        map_rows.append({
            "MAP NUMBER": values.map_number,
            "FILENAME": filename,
        })

        # --------------------------------------------------
        # Sheet 2: Wind data
        # --------------------------------------------------
        if len(values.wind_real) == 0:
            continue
        wind_row = {
            "MAP NUMBER": values.map_number,
            "ROW DB": values.row_db,
            # "STABILITY": values.stability,
            "L REAL [m]": values.L_real,
            # "L RETRIEVED [m]": values.L_retrieved,
            # "U* RETRIEVED [m/s]": values.u_star_retrieved,
            "REAL MC WIND [42 m]": values.wind_real[0],
            "REAL DIRECTION [deg]": values.wind_direction,
        }
        len_wind_row = len(wind_row)
        # --------------------------------------------------
        # Sheet 3: Atmospheric data
        # --------------------------------------------------
        atmospheric_rows.append({
            "MAP NUMBER": values.map_number,
            "ROW DB": values.row_db,
            "STABILITY": values.stability,
            "L REAL [m]": values.L_real,
            "L RETRIEVED [m]": values.L_retrieved,
            "U* RETRIEVED [m/s]": values.u_star_retrieved,
        })
        # --------------------------------------------------
        # Sheet 4: Stadistics Wind data
        # --------------------------------------------------
        abserr_row = {
            "MAP NUMBER": values.map_number,
        }
        relerr_row = {
            "MAP NUMBER": values.map_number,
        }

        # Retrieved wind for each level
        len_levels = 0
        if values.wind_retrieved_10m is not None:
            for level, wind in values.wind_retrieved_10m.items():
                wind_real = values.wind_real[0]
                error = wind - wind_real
                wind_row[f"Level [{level}]"] = wind
                abserr_row[f"{level}"] = error
                relerr_row[f"{level}"] = abs(error / wind_real)
                len_levels += 1
                if values.map_number in avoid_for_metrics:
                    continue
                errors_by_level.setdefault(level, []).append(error)
                

        wind_rows.append(wind_row)
        abserr_rows.append(abserr_row)
        relerr_rows.append(relerr_row)
    
    metrics_by_level = {}
    for level, errors in errors_by_level.items():
        metrics_by_level[level] = {
            "MAE": np.mean(np.abs(errors)),
            "RMSE": np.sqrt(np.mean(np.square(errors))),
            "BIAS": np.mean(errors),
        }
    
    # Build dataframes
    map_df = pd.DataFrame(map_rows, columns=["MAP NUMBER", "FILENAME"])
    wind_df = pd.DataFrame(wind_rows)
    atmospheric_df = pd.DataFrame(atmospheric_rows)
    abserr_df = pd.DataFrame(abserr_rows)
    relerr_df = pd.DataFrame(relerr_rows)
    metrics_df = pd.DataFrame(metrics_by_level)
    metrics_df.index.name = "MAP NUMBER"
    metrics_df = metrics_df.reset_index()
    abserr_df.columns = abserr_df.columns.map(str)
    relerr_df.columns = relerr_df.columns.map(str)
    metrics_df.columns = metrics_df.columns.map(str)
    metrics_df = metrics_df.rename(columns={"index": "MAP NUMBER"})
    empty_rows = pd.DataFrame([{},{}])
    stadistics_wind_df = pd.concat([abserr_df, empty_rows, relerr_df, empty_rows, metrics_df], ignore_index=True, sort=False)

    # Export both sheets
    light_grey = PatternFill(fill_type="solid", fgColor="D9D9D9")
    thin_grey = Side(style="thin", color="808080")
    title_border = Border(left=thin_grey, right=thin_grey, top=thin_grey, bottom=thin_grey)
    title_alignment = Alignment(horizontal="center", vertical="center")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Map sheet
        map_df.to_excel(writer, sheet_name="map", index=False, startrow=1)
        ms = writer.sheets["map"]
        ms.merge_cells("A1:B1")
        ms["A1"] = "Map"
        ms["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ms["A1"].fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
        # Wind sheet
        wind_df.to_excel(writer, sheet_name="Wind retrieved", index=False, startrow=1)
        ws = writer.sheets["Wind retrieved"]
        start_ws_1 = get_column_letter(len_wind_row)
        start_ws_2 = get_column_letter(len_wind_row+1)
        end_ws = get_column_letter(len_wind_row + len_levels + 1)
        ws.merge_cells(f"A1:{start_ws_1}1")
        ws.merge_cells(f"{start_ws_2}1:{end_ws}1")
        ws["A1"] = "Initial parameters"
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws[f"{start_ws_2}1"] = "Levels"
        ws[f"{start_ws_2}1"].alignment = Alignment(horizontal="center", vertical="center")
        
        atmospheric_df.to_excel(writer, sheet_name="Atmospheric retrieved", index=False, startrow=1)
        stadistics_wind_df.to_excel(writer, sheet_name="Statistics Wind", index=False, startrow=1)
        
        for sheet, ranges in [
            (ms, ["A1:B1"]),
            (ws, [f"A1:{start_ws_1}1", f"{start_ws_2}1:{end_ws}1"]),
        ]:
            for cell_range in ranges:
                for row in sheet[cell_range]:
                    for cell in row:
                        cell.fill = light_grey
                        cell.border = title_border
                        cell.alignment = title_alignment

        # Optional: title row height
        ms.row_dimensions[1].height = 25
        ws.row_dimensions[1].height = 25

    print(f"Excel file generated: {output_path}")


def postprocess_statistics_wind(
    output_path="TFM_results.xlsx",
    sheet_name="Statistics Wind",
):
    """
    Post-process the exported Excel file.

    In the 'Statistics Wind' sheet, apply a red-to-green color scale
    independently to the MAE, RMSE and BIAS rows.

    Lower values are green; higher values are red.
    Only numeric metric cells are considered.
    """

    wb = load_workbook(output_path)

    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(
            f"Sheet '{sheet_name}' not found in '{output_path}'."
        )

    ws = wb[sheet_name]

    target_metrics = {"MAE", "RMSE", "BIAS"}

    # Locate the metric rows using the first column.
    for row in ws.iter_rows(min_row=1):
        metric = row[0].value

        if not isinstance(metric, str):
            continue

        metric = metric.strip().upper()

        if metric not in target_metrics:
            continue

        # Collect numeric cells in the metric row, excluding column A.
        numeric_cells = [
            cell
            for cell in row[1:]
            if isinstance(cell.value, (int, float))
            and not isinstance(cell.value, bool)
        ]

        if not numeric_cells:
            continue

        # Calculate the range independently for this metric.
        min_value = min(cell.value for cell in numeric_cells)
        max_value = max(cell.value for cell in numeric_cells)

        # If all values are equal, use a neutral color.
        if min_value == max_value:
            neutral_fill = PatternFill(
                fill_type="solid",
                fgColor="FFFF00",
            )
            for cell in numeric_cells:
                cell.fill = neutral_fill
            continue

        # Apply a red -> yellow -> green scale.
        first_col = numeric_cells[0].column_letter
        last_col = numeric_cells[-1].column_letter
        row_number = numeric_cells[0].row

        color_rule = ColorScaleRule(
            start_type="min",
            start_color="F8696B",  # Red: highest error
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",    # Yellow: intermediate
            end_type="max",
            end_color="63BE7B",    # Green: lowest error
        )

        # Excel's color scale maps min -> red and max -> green.
        # Reverse the endpoints so lower values are green.
        color_rule = ColorScaleRule(
            start_type="min",
            start_color="63BE7B",  # Green: lowest value
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="F8696B",    # Red: highest value
        )

        ws.conditional_formatting.add(
            f"{first_col}{row_number}:{last_col}{row_number}",
            color_rule,
        )

    wb.save(output_path)
    wb.close()

    print(f"Statistics color formatting applied: {output_path}")