import pandas as pd
import numpy as np
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from copy import copy
from TFM_utils.data_retrieve import Data_Retrieved

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
    
    fin_relerr_rows = []
    fin_abserr_rows = []
    fin_errors_by_level = {}

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
            "REAL WIND EXTRAPOLATED [10m]": values.wind_new_value_10m,
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
        # --------------------------------------------------
        # Sheet 5: Final Stadistics Wind data
        # --------------------------------------------------
        fin_abserr_row = {
            "MAP NUMBER": values.map_number,
        }
        fin_relerr_row = {
            "MAP NUMBER": values.map_number,
        }

        # Retrieved wind for each level
        len_levels = 0
        if values.wind_retrieved_10m is not None:
            for level, wind in values.wind_retrieved_10m.items():
                wind_real = values.wind_real[0]
                final_wind = values.wind_new_value_10m
                error = wind - wind_real # Wind real at certain heigth
                final_error = wind- final_wind # Wind extrapolated at 10m
                wind_row[f"Level [{level}]"] = wind
                abserr_row[f"{level}"] = error
                relerr_row[f"{level}"] = abs(error / wind_real)
                fin_abserr_row[f"{level}"] = final_error
                fin_relerr_row[f"{level}"] = abs(final_error / final_wind)
                len_levels += 1
                if values.map_number in avoid_for_metrics:
                    continue
                errors_by_level.setdefault(level, []).append(error)
                fin_errors_by_level.setdefault(level, []).append(final_error)
                

        wind_rows.append(wind_row)
        abserr_rows.append(abserr_row)
        relerr_rows.append(relerr_row)
        
        fin_abserr_rows.append(fin_abserr_row)
        fin_relerr_rows.append(fin_relerr_row)
    
    metrics_by_level = {}
    for level, errors in errors_by_level.items():
        metrics_by_level[level] = {
            "MAE": np.mean(np.abs(errors)),
            "RMSE": np.sqrt(np.mean(np.square(errors))),
            "BIAS": np.mean(errors),
        }
        
    final_metrics_by_level = {}
    for level, errors in fin_errors_by_level.items():
        final_metrics_by_level[level] = {
            "MAE": np.mean(np.abs(errors)),
            "RMSE": np.sqrt(np.mean(np.square(errors))),
            "BIAS": np.mean(errors),
        }
    
    # Build dataframes
    map_df = pd.DataFrame(map_rows, columns=["MAP NUMBER", "FILENAME"])
    wind_df = pd.DataFrame(wind_rows)
    atmospheric_df = pd.DataFrame(atmospheric_rows)
    # --------------------------------------------
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
    # ------------------------------------------
    fin_abserr_df = pd.DataFrame(fin_abserr_rows)
    fin_relerr_df = pd.DataFrame(fin_relerr_rows)
    fin_metrics_df = pd.DataFrame(final_metrics_by_level)
    fin_metrics_df.index.name = "MAP NUMBER"
    fin_metrics_df = fin_metrics_df.reset_index()
    fin_abserr_df.columns = fin_abserr_df.columns.map(str)
    fin_relerr_df.columns = fin_relerr_df.columns.map(str)
    fin_metrics_df.columns = fin_metrics_df.columns.map(str)
    fin_metrics_df = fin_metrics_df.rename(columns={"index": "MAP NUMBER"})
    fin_stadistics_wind_df = pd.concat([fin_abserr_df, empty_rows, fin_relerr_df, empty_rows, fin_metrics_df], ignore_index=True, sort=False)

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
        fin_stadistics_wind_df.to_excel(writer, sheet_name="Final Statistics Wind", index=False, startrow=1)
        
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
    wb = load_workbook(output_path)

    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(
            f"Sheet '{sheet_name}' not found in '{output_path}'."
        )

    ws = wb[sheet_name]
    target_metrics = {"MAE", "RMSE", "BIAS"}

    # Identify metric rows before inserting any new rows.
    metric_rows = []

    for row in range(1, ws.max_row + 1):
        value = ws.cell(row=row, column=1).value

        if isinstance(value, str):
            metric = value.strip().upper()

            if metric in target_metrics:
                metric_rows.append((row, metric))

    # Process from bottom to top so inserted rows do not
    # change the positions of metrics still to be processed.
    for row_number, metric in reversed(metric_rows):

        # Collect numeric cells, excluding column A.
        numeric_cells = [
            ws.cell(row=row_number, column=col)
            for col in range(2, ws.max_column + 1)
            if isinstance(
                ws.cell(row=row_number, column=col).value,
                (int, float),
            )
            and not isinstance(
                ws.cell(row=row_number, column=col).value,
                bool,
            )
        ]

        if not numeric_cells:
            continue

        # For BIAS, insert an auxiliary row containing
        # absolute values. Keep the original BIAS unchanged.
        if metric == "BIAS":
            aux_row = row_number + 1
            ws.insert_rows(aux_row)

            # Copy basic formatting from the original row.
            for col in range(1, ws.max_column + 1):
                source = ws.cell(row=row_number, column=col)
                target = ws.cell(row=aux_row, column=col)

                if source.has_style:
                    target._style = copy(source._style)

                if source.number_format:
                    target.number_format = source.number_format

                target.alignment = copy(source.alignment)
                target.protection = copy(source.protection)

                if source.hyperlink:
                    target._hyperlink = copy(source.hyperlink)

            ws.cell(aux_row, 1).value = "BIAS (ABS.)"

            for cell in numeric_cells:
                ws.cell(
                    row=aux_row,
                    column=cell.column,
                ).value = abs(cell.value)

            # Apply colors to the auxiliary row, not the
            # original BIAS row.
            cells_to_color = [
                ws.cell(row=aux_row, column=cell.column)
                for cell in numeric_cells
            ]

        else:
            # MAE and RMSE use their original values.
            cells_to_color = numeric_cells

        values = [cell.value for cell in cells_to_color]
        min_value = min(values)
        max_value = max(values)

        # If all values are equal, use a neutral color.
        if min_value == max_value:
            neutral_fill = PatternFill(
                fill_type="solid",
                fgColor="FFEB84",
            )

            for cell in cells_to_color:
                cell.fill = neutral_fill

            continue

        # Apply a green -> yellow -> red color scale.
        first_col = cells_to_color[0].column_letter
        last_col = cells_to_color[-1].column_letter
        color_row = cells_to_color[0].row

        color_rule = ColorScaleRule(
            start_type="num",
            start_value=min_value,
            start_color="63BE7B",  # Green: lowest error
            mid_type="num",
            mid_value=(min_value + max_value) / 2,
            mid_color="FFEB84",    # Yellow: intermediate
            end_type="num",
            end_value=max_value,
            end_color="F8696B",    # Red: highest error
        )

        ws.conditional_formatting.add(
            f"{first_col}{color_row}:{last_col}{color_row}",
            color_rule,
        )

    wb.save(output_path)
    wb.close()

    print(f"Statistics color formatting applied: {output_path}")