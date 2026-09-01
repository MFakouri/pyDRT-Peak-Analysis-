# -*- coding: utf-8 -*-
"""Graphical user interface for DRT peak deconvolution and analysis."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from numpy import absolute, angle

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from . import layout
from . import basics
from .runs import EIS_object, simple_run, Bayesian_run, BHT_run, peak_analysis
from .importer import read_eis_file, apply_active_area
from .export_utils import export_drt_csv, export_eis_csv, export_parameters_csv, compute_peak_summary
from .cnls_peak_fit import fit_with_frequency_bounds


class ActiveAreaDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Active Area")
        self.setModal(True)
        self.resize(470, 210)
        v = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Active Area Correction")
        f = title.font(); f.setBold(True); title.setFont(f)
        v.addWidget(title)
        info = QtWidgets.QLabel(
            "Check the box only if active area has NOT already been applied to the impedance."
        )
        info.setWordWrap(True)
        v.addWidget(info)
        self.check = QtWidgets.QCheckBox("Active area has NOT already been applied")
        v.addWidget(self.check)
        row = QtWidgets.QHBoxLayout()
        self.area_label = QtWidgets.QLabel("Active area (cm²):")
        self.area_edit = QtWidgets.QLineEdit()
        self.area_edit.setPlaceholderText("e.g. 5.0")
        row.addWidget(self.area_label); row.addWidget(self.area_edit)
        v.addLayout(row)
        self.area_label.setEnabled(False); self.area_edit.setEnabled(False)
        self.check.toggled.connect(self.area_label.setEnabled)
        self.check.toggled.connect(self.area_edit.setEnabled)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def _accept(self):
        if self.check.isChecked():
            try:
                value = float(self.area_edit.text().strip())
            except ValueError:
                value = np.nan
            if not np.isfinite(value) or value <= 0:
                QtWidgets.QMessageBox.critical(self, "Active Area Error", "Please enter a valid positive Active area (cm²).")
                return
        self.accept()

    @property
    def apply_area(self):
        return self.check.isChecked()

    @property
    def area(self):
        return float(self.area_edit.text()) if self.apply_area else 1.0


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = layout.Ui_MainWindow()
        self.ui.setupUi(self)
        # Software attribution and citation
        self.credit_label = QtWidgets.QLabel()

        self.credit_label.setText(
          'Based on <a href="https://github.com/ciuccislab/pyDRTtools">'
          'pyDRTtools</a> (Ciucci Lab) &nbsp; | &nbsp; '
          'Modified and extended by Masood Fakouri Hasanabadi &nbsp; | &nbsp; '
          'Please cite both the original work and '
          '<a href="https://github.com/MFakouri/pyDRT-Peak-Analysis-">'
          'this software</a>.'
        )

        self.credit_label.setOpenExternalLinks(True)
        self.credit_label.setTextFormat(QtCore.Qt.RichText)
        self.credit_label.setStyleSheet(
          "QLabel {"
          "font-size: 9pt;"
          "color: #555555;"
          "padding: 2px 6px;"
          "}"
        )

        self.ui.statusbar.addPermanentWidget(self.credit_label)
        self.data = None

        # Default analysis settings
        self.ui.discre_choice.setCurrentIndex(0)       # Gaussian
        self.ui.data_used_choice.setCurrentIndex(0)    # Combined Re-Im
        self.ui.induct_choice.setCurrentIndex(1)       # Fitting with Inductance
        self.ui.der_choice.setCurrentIndex(1)          # 2nd order
        self.ui.lambda_choice.setCurrentIndex(0)       # custom
        self.ui.reg_param_entry.setText("0.001")
        self.ui.reg_param_label.setText("Custom Regularization parameter")
        self.ui.reg_param_entry_2.setReadOnly(True)
        self.ui.FWHM_entry.setText("0.5")
        self.ui.shape_control_choice.setCurrentIndex(0)
        self.ui.sample_no_entry.setText("2000")

        # Peak-parameter table below the plot. It is shown only after
        # a successful peak deconvolution.
        self.peak_table_label = QtWidgets.QLabel("Peak Parameters", self.ui.frame)
        self.peak_table_label.setGeometry(QtCore.QRect(5, 731, 180, 18))
        peak_label_font = self.peak_table_label.font()
        peak_label_font.setBold(True)
        self.peak_table_label.setFont(peak_label_font)
        self.peak_table_label.hide()

        self.peak_table = QtWidgets.QTableWidget(self.ui.frame)
        self.peak_table.setGeometry(QtCore.QRect(0, 750, 901, 191))
        self.peak_table.setColumnCount(7)
        self.peak_table.setHorizontalHeaderLabels([
            "Series", "Area (Ω·cm²)", "R (Ω·cm²)",
            "Tau (s)", "C (F/cm²)", "Frequency (Hz)", "FWHM ln(Tau)"
        ])
        self.peak_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peak_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.peak_table.setAlternatingRowColors(True)
        self.peak_table.verticalHeader().setVisible(False)
        self.peak_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.peak_table.setStyleSheet("QTableWidget { font-size: 10pt; }")
        self.peak_table.hide()

        # CNLS-fit table used only by the EIS Score view. The Score view has
        # been repurposed from BHT score bars to input-vs-fitted Nyquist data.
        self.cnls_table_label = QtWidgets.QLabel("CNLS Fit Parameters", self.ui.frame)
        self.cnls_table_label.setGeometry(QtCore.QRect(5, 641, 220, 18))
        cnls_label_font = self.cnls_table_label.font()
        cnls_label_font.setBold(True)
        self.cnls_table_label.setFont(cnls_label_font)
        self.cnls_table_label.hide()

        self.cnls_table = QtWidgets.QTableWidget(self.ui.frame)
        self.cnls_table.setGeometry(QtCore.QRect(0, 660, 901, 281))
        self.cnls_table.setColumnCount(6)
        self.cnls_table.setHorizontalHeaderLabels([
            "Component", "Value", "R (Ω·cm²)", "C (F/cm²)", "Frequency (Hz)", "Tau (s)"
        ])
        self.cnls_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cnls_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cnls_table.setAlternatingRowColors(True)
        self.cnls_table.verticalHeader().setVisible(False)
        self.cnls_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.cnls_table.setStyleSheet("QTableWidget { font-size: 10pt; }")
        self.cnls_table.hide()

        # Configure DRT plot and export modes.
        self.ui.peak_method_label.setText("DRT plot")
        self.ui.peak_method_choice.clear()
        self.ui.peak_method_choice.addItems([
            "Gamma vs Tau", "Gamma vs Frequency", "g vs Tau", "g vs Frequency"
        ])
        self.ui.peak_method_choice.setCurrentIndex(0)
        self.ui.peak_method_choice.currentIndexChanged.connect(
            lambda: self.plotting_callback('DRT_data') if self.data is not None and self.data.method != 'none' else None
        )
        self.ui.lambda_choice.currentIndexChanged.connect(self.lambda_method_callback)
        self.lambda_method_callback()

        self.ui.import_button.clicked.connect(self.import_file)
        self.ui.induct_choice.currentIndexChanged.connect(self.inductance_callback)
        self.ui.show_EIS.clicked.connect(lambda: self.plotting_callback('EIS_data'))
        self.ui.show_mag.clicked.connect(lambda: self.plotting_callback('Magnitude'))
        self.ui.show_phase.clicked.connect(lambda: self.plotting_callback('Phase'))
        self.ui.show_re.clicked.connect(lambda: self.plotting_callback('Re_data'))
        self.ui.show_im.clicked.connect(lambda: self.plotting_callback('Im_data'))
        self.ui.show_re_res.clicked.connect(lambda: self.plotting_callback('Re_residual'))
        self.ui.show_im_res.clicked.connect(lambda: self.plotting_callback('Im_residual'))
        self.ui.show_DRT.clicked.connect(lambda: self.plotting_callback('DRT_data'))
        self.ui.show_score.clicked.connect(lambda: self.plotting_callback('Score'))
        self.ui.simple_run_button.clicked.connect(self.simple_run_callback)
        self.ui.bayesian_button.clicked.connect(self.bayesian_run_callback)
        self.ui.HT_button.clicked.connect(self.BHT_run_callback)
        self.ui.peak_decon_button.clicked.connect(self.peak_analysis_run_callback)
        self.ui.export_DRT_button.clicked.connect(self.export_DRT)
        self.ui.export_EIS_button.clicked.connect(self.export_EIS)
        self.ui.export_fig_button.clicked.connect(self.export_fig)

    def lambda_method_callback(self):
        """Enable the custom lambda box only when the custom method is selected."""
        is_custom = str(self.ui.lambda_choice.currentText()) == 'custom'
        self.ui.reg_param_entry.setEnabled(is_custom)
        if is_custom:
            self.ui.reg_param_label.setText("Custom Regularization parameter")
        else:
            self.ui.reg_param_label.setText("Custom lambda (not used for auto methods)")

    def _error(self, title, exc):
        QtWidgets.QMessageBox.critical(self, title, str(exc))

    def _hide_peak_table(self):
        self.peak_table.clearContents()
        self.peak_table.setRowCount(0)
        self.peak_table.hide()
        self.peak_table_label.hide()
        if not self.cnls_table.isVisible():
            self.ui.plot_panel.setGeometry(QtCore.QRect(0, 40, 901, 901))

    def _hide_cnls_table(self):
        self.cnls_table.clearContents()
        self.cnls_table.setRowCount(0)
        self.cnls_table.hide()
        self.cnls_table_label.hide()
        if not self.peak_table.isVisible():
            self.ui.plot_panel.setGeometry(QtCore.QRect(0, 40, 901, 901))

    def _show_cnls_table(self):
        fit = getattr(self.data, 'cnls_fit', None) if self.data is not None else None
        if not fit:
            self._hide_cnls_table()
            return

        rows = [
            ("L (H·cm²)", fit['L'], None, None, None, None),
            ("R_ohmic (Ω·cm²)", fit['R'], None, None, None, None),
        ]
        for rc in fit['rc']:
            rows.append((
                rc['component'], None, rc['R'], rc['C'], rc['frequency'], rc['tau']
            ))

        self.cnls_table.setRowCount(len(rows))
        for row_idx, values in enumerate(rows):
            for col_idx, value in enumerate(values):
                if value is None:
                    text = ""
                elif col_idx == 0:
                    text = str(value)
                else:
                    text = f"{float(value):.6g}"
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.cnls_table.setItem(row_idx, col_idx, item)

        self._hide_peak_table()
        self.ui.plot_panel.setGeometry(QtCore.QRect(0, 40, 901, 601))
        self.cnls_table_label.show()
        self.cnls_table.show()

    def _show_peak_table(self):
        self._hide_cnls_table()
        df = compute_peak_summary(self.data)
        if df.empty:
            self._hide_peak_table()
            return

        columns = [
            'Series_Name', 'Area_Ohm_cm2', 'R_Ohm_cm2',
            'Tau_s', 'Capacitance_F_per_cm2', 'Frequency_Hz', 'FWHM_lnTau'
        ]

        self.peak_table.setRowCount(len(df))


        for row_idx, (_, row) in enumerate(df[columns].iterrows()):
            for col_idx, column in enumerate(columns):
                value = row[column]

                if column == 'Series_Name':
                    if str(value) == 'Total Sum':
                        text = 'Total Sum'
                    else:
                        text = f"Peak_{row_idx + 1}"

                elif np.isfinite(value):
                    text = f"{float(value):.6g}"
                else:
                    text = ""

                item = QtWidgets.QTableWidgetItem(text)
                if column not in {'Series_Name'}:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.peak_table.setItem(row_idx, col_idx, item)

        self.ui.plot_panel.setGeometry(QtCore.QRect(0, 40, 901, 691))
        self.peak_table_label.show()
        self.peak_table.show()

    def _settings(self):
        return dict(
            rbf_type=str(self.ui.discre_choice.currentText()),
            data_used=str(self.ui.data_used_choice.currentText()),
            induct_used=int(self.ui.induct_choice.currentIndex()),
            der_used=str(self.ui.der_choice.currentText()),
            cv_type=str(self.ui.lambda_choice.currentText()),
            reg_param=float(self.ui.reg_param_entry.text()),
            shape_control=str(self.ui.shape_control_choice.currentText()),
            coeff=float(self.ui.FWHM_entry.text()),
        )

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an EIS file", "",
            "EIS data (*.mat *.txt *.csv *.z *.dat *.mpr *.dta *.ism);;All Files (*)"
        )
        if not path:
            return
        try:
            arr = read_eis_file(path)
            dlg = ActiveAreaDialog(self)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            if dlg.apply_area:
                arr = apply_active_area(arr, dlg.area)

            self.data = EIS_object(arr[:, 0], arr[:, 1], arr[:, 2])
            self.data.original_input_name = Path(path).stem
            self.data.source_path = path
            self.data.active_area_applied_here = dlg.apply_area
            self.data.active_area_cm2 = dlg.area
            self.data.cnls_fit = None
            self.inductance_callback()
            self.statusBar().showMessage(f"Imported file: {path}", 3000)
        except Exception as exc:
            self._error("EIS Import Error", exc)

    def inductance_callback(self):
        if self.data is None:
            return
        try:
            if self.ui.induct_choice.currentIndex() == 2:
                # MATLAB: discard rows for which -Z'' < 0, i.e. retain Z'' <= 0.
                mask = self.data.Z_double_prime_0 <= 0
                self.data.freq = self.data.freq_0[mask].copy()
                self.data.Z_prime = self.data.Z_prime_0[mask].copy()
                self.data.Z_double_prime = self.data.Z_double_prime_0[mask].copy()
                self.data.Z_exp = self.data.Z_prime + 1j*self.data.Z_double_prime
            else:
                self.data.freq = self.data.freq_0.copy()
                self.data.Z_prime = self.data.Z_prime_0.copy()
                self.data.Z_double_prime = self.data.Z_double_prime_0.copy()
                self.data.Z_exp = self.data.Z_exp_0.copy()
            self.data.tau = 1.0/self.data.freq
            self.data._reset_tau_fine()
            self.data.method = 'none'
            self.data.cnls_fit = None
            self._hide_peak_table()
            self._hide_cnls_table()
            self.plotting_callback('EIS_data')
        except Exception as exc:
            self._error("Inductance Error", exc)

    def simple_run_callback(self):
        if self.data is None:
            return
        try:
            settings = self._settings()
            self.data = simple_run(self.data, **settings)
            lambda_value = float(getattr(self.data, 'lambda_value', settings['reg_param']))
            self.ui.reg_param_entry_2.setText(f"{lambda_value:.10g}")

            if settings.get('cv_type') == 'custom':
                self.ui.reg_param_label_2.setText("Regularization parameter used")
            else:
                diagnostics = getattr(basics.optimal_lambda, 'last_diagnostics', None)
                self.data.lambda_diagnostics = diagnostics
                if diagnostics and diagnostics.get('fallback_used'):
                    requested = diagnostics.get('requested_method', settings['cv_type'])
                    effective = diagnostics.get('effective_method', 'LC')
                    primary_lambda = diagnostics['primary']['lambda_value']
                    self.ui.reg_param_label_2.setText(
                        f"Selected λ ({effective} fallback)"
                    )
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No interior optimum for requested criterion",
                        f"{requested} decreases to the search boundary (λ = {primary_lambda:.3g}), "
                        "so it does not define a reliable interior optimum for this spectrum and "
                        "the current DRT settings.\n\n"
                        f"The program used the interior {effective} solution instead: "
                        f"λ = {lambda_value:.6g}.\n\n"
                        "The requested method and fallback are both reported explicitly; the "
                        "boundary value is not labelled as an optimal λ."
                    )
                elif diagnostics and diagnostics.get('boundary_hit'):
                    self.ui.reg_param_label_2.setText("Boundary λ (no interior optimum)")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Regularization selection warning",
                        f"{settings['cv_type']} reached a search boundary at λ = {lambda_value:.3g}.\n\n"
                        "No reliable interior optimum was found and no valid fallback was available."
                    )
                else:
                    self.ui.reg_param_label_2.setText(
                        f"Optimal Regularization parameter ({settings['cv_type']})"
                    )
            self.data.cnls_fit = None
            self._hide_peak_table()
            self._hide_cnls_table()
            self.plotting_callback('DRT_data')
        except Exception as exc:
            self._error("Simple Run Error", exc)

    def bayesian_run_callback(self):
        if self.data is None:
            return
        try:
            settings = self._settings()
            settings['NMC_sample'] = int(self.ui.sample_no_entry.text())
            self.data = Bayesian_run(self.data, **settings)
            self.data.cnls_fit = None
            self._hide_peak_table()
            self._hide_cnls_table()
            self.plotting_callback('DRT_data')
        except Exception as exc:
            self._error("Bayesian Run Error", exc)

    def BHT_run_callback(self):
        if self.data is None:
            return
        try:
            self.data = BHT_run(
                self.data,
                str(self.ui.discre_choice.currentText()),
                str(self.ui.der_choice.currentText()),
                str(self.ui.shape_control_choice.currentText()),
                float(self.ui.FWHM_entry.text()),
            )
            self.data.cnls_fit = None
            self._hide_peak_table()
            self._hide_cnls_table()
            self.plotting_callback('DRT_data')
        except Exception as exc:
            self._error("BHT Error", exc)

    def peak_analysis_run_callback(self):
        if self.data is None:
            return
        try:
            n_peaks = max(1, int(round(abs(float(self.ui.peak_num_entry.text())))))
            frequency_range_scale = float(self.ui.freq_range_scale_entry.value())
            settings = self._settings()
            self.data = peak_analysis(
                self.data, **settings, peak_method='separate',
                N_peaks=n_peaks
            )

            # The requested number of peaks is also the number of RC branches.
            # Initial R/C/f come directly from peak deconvolution in
            # high-frequency -> low-frequency order. Frequency is then allowed
            # to move only inside the frequency-dependent bounds scaled by the
            # user-selected Frequency Range Scale (1.0 preserves the default).
            fit = fit_with_frequency_bounds(
                self.data,
                expected_n_peaks=n_peaks,
                frequency_range_scale=frequency_range_scale,
            )

            self.plotting_callback('DRT_data')
            self._show_peak_table()
            self.statusBar().showMessage(
                f"Peak deconvolution complete: {n_peaks} RC branches. "
                f"CNLS fitted L={fit['L']:.4g}, series R={fit['R']:.4g}, and RC values; "
                "peak frequencies were fitted within their log-domain bounds. "
                "Open RCs fitting to view the Nyquist fit and parameter table.",
                8000
            )
        except Exception as exc:
            self._error("Peak Analysis / CNLS Fit Error", exc)

    def plotting_callback(self, plot_to_show):
        if self.data is None:
            return

        # The EIS Score view is reserved exclusively for the bounded-frequency CNLS
        # Nyquist fit and its parameter table. Other views restore their own
        # normal table/plot layout.
        if plot_to_show == 'Score':
            self._hide_peak_table()
            self._show_cnls_table()
        elif plot_to_show == 'DRT_data' and getattr(self.data, 'method', '') == 'peak':
            self._hide_cnls_table()
            self._show_peak_table()
        else:
            self._hide_peak_table()
            self._hide_cnls_table()

        fig = Figure_Canvas()
        func = getattr(fig, plot_to_show)
        if plot_to_show == 'DRT_data':
            func(self.data, str(self.ui.peak_method_choice.currentText()))
        else:
            func(self.data)
        scene = QtWidgets.QGraphicsScene(20, 25, 650, 610)
        scene.addWidget(fig)
        self.ui.plot_panel.setScene(scene)
        self.ui.plot_panel.show()
        self._current_canvas = fig

    def _choose_output_folder(self, title):
        start = str(Path(self.data.source_path).parent) if self.data is not None and self.data.source_path else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, title, start)
        return Path(folder) if folder else None

    def export_DRT(self):
        if self.data is None or self.data.method == 'none':
            return
        folder = self._choose_output_folder("Choose folder for DRT data")
        if folder is None:
            return
        try:
            name = self.data.original_input_name or 'EIS'
            path = folder/f"{name} DRT.csv"
            export_drt_csv(self.data, path, str(self.ui.peak_method_choice.currentText()))
            self.statusBar().showMessage(f"Saved: {path}", 4000)
        except Exception as exc:
            self._error("DRT Export Error", exc)

    def export_EIS(self):
        if self.data is None or self.data.method == 'none':
            return
        folder = self._choose_output_folder("Choose folder for EIS comparison")
        if folder is None:
            return
        try:
            name = self.data.original_input_name or 'EIS'
            path = folder/f"{name} EIS comparison.csv"
            export_eis_csv(self.data, path)
            self.statusBar().showMessage(f"Saved: {path}", 4000)
        except Exception as exc:
            self._error("EIS Export Error", exc)

    def export_fig(self):
        if self.data is None or self.data.method == 'none':
            return
        folder = self._choose_output_folder("Choose folder for figures and parameters")
        if folder is None:
            return
        try:
            name = self.data.original_input_name or 'EIS'

            # Save the standard DRT (Gamma vs Tau) figure.
            drt_path = folder/f"{name} DRT.png"
            drt_fig = Figure_Canvas()
            drt_fig.DRT_data(self.data, 'Gamma vs Tau')
            drt_fig.figure.savefig(drt_path, dpi=300, bbox_inches='tight')

            # Save the RCs fitting (Nyquist/CNLS) figure.
            rcs_path = folder/f"{name} RCs fitting.png"
            rcs_fig = Figure_Canvas()
            rcs_fig.Score(self.data)
            rcs_fig.figure.savefig(rcs_path, dpi=300, bbox_inches='tight')

            # Save the combined parameters CSV using the current export layout.
            params = export_parameters_csv(self.data, folder, name)

            saved_names = [drt_path.name, rcs_path.name]
            if params is not None:
                saved_names.append(params.name)
            self.statusBar().showMessage(
                "Saved: " + ", ".join(saved_names), 5000
            )
        except Exception as exc:
            self._error("Figure Export Error", exc)


class Figure_Canvas(FigureCanvas):
    def __init__(self, parent=None, width=7.5, height=6.5, dpi=100):
        plt.ioff()
        plt.rc('font', family='serif', size=20)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)
        self.figure = fig
        fig.tight_layout()
        # Adjust plot position for improved spacing.
        fig.subplots_adjust(left=0.18, bottom=0.15, right=0.94, top=0.93)

    def _freq_axis(self, entry):
        self.axes.set_xscale('log')
        self.axes.set_xlim(float(np.max(entry.freq_0)), float(np.min(entry.freq_0)))
        self.axes.tick_params(axis='x', which='both', direction='inout')

    def EIS_data(self, entry):
        if entry.method == 'BHT':
            self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im, 'k', label='$Z_\\mu$(Regressed)', linewidth=3)
            self.axes.plot(entry.mu_Z_H_re_agm, -entry.mu_Z_H_im_agm, 'b', label='$Z_H$(Hilbert transform)', linewidth=3)
            self.axes.plot(entry.Z_prime, -entry.Z_double_prime, 'or', markersize=4)
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')
        elif entry.method != 'none':
            self.axes.plot(entry.mu_Z_re, -entry.mu_Z_im, 'k', linewidth=3)
            self.axes.plot(entry.Z_prime, -entry.Z_double_prime, 'or', markersize=4)
        else:
            self.axes.plot(entry.Z_prime, -entry.Z_double_prime, 'or', markersize=4)
        self.axes.set_xlabel(r"$Z^{\prime}/(\Omega\,\mathrm{cm}^2)$")
        self.axes.set_ylabel(r"$-Z^{\prime\prime}/(\Omega\,\mathrm{cm}^2)$")
        self.axes.axis('equal')

    def Magnitude(self, entry):
        if entry.method == 'BHT':
            self.axes.plot(entry.freq, absolute(entry.mu_Z_re+1j*entry.mu_Z_im), 'k', linewidth=3, label='$Z_\\mu$(Regressed)')
            self.axes.plot(entry.freq, absolute(entry.mu_Z_H_re_agm+1j*entry.mu_Z_H_im_agm), 'b', linewidth=3, label='$Z_H$(Hilbert transform)')
            self.axes.plot(entry.freq, absolute(entry.Z_exp), 'or', markersize=4)
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')
        elif entry.method != 'none':
            self.axes.plot(entry.freq, absolute(entry.mu_Z_re+1j*entry.mu_Z_im), 'k', linewidth=3)
            self.axes.plot(entry.freq, absolute(entry.Z_exp), 'or', markersize=4)
        else:
            self.axes.plot(entry.freq, absolute(entry.Z_exp), 'or', markersize=4)
        self._freq_axis(entry)
        self.axes.set_xlabel('$f$/Hz')
        self.axes.set_ylabel(r'$|Z|/(\Omega\,\mathrm{cm}^2)$')

    def Phase(self, entry):
        if entry.method == 'BHT':
            self.axes.plot(entry.freq, angle(entry.mu_Z_re+1j*entry.mu_Z_im, deg=True), 'k', linewidth=3, label='$Z_\\mu$(Regressed)')
            self.axes.plot(entry.freq, angle(entry.mu_Z_H_re_agm+1j*entry.mu_Z_H_im_agm, deg=True), 'b', linewidth=3, label='$Z_H$(Hilbert transform)')
            self.axes.plot(entry.freq, angle(entry.Z_exp, deg=True), 'or', markersize=4)
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')
        elif entry.method != 'none':
            self.axes.plot(entry.freq, angle(entry.mu_Z_re+1j*entry.mu_Z_im, deg=True), 'k', linewidth=3)
            self.axes.plot(entry.freq, angle(entry.Z_exp, deg=True), 'or', markersize=4)
        else:
            self.axes.plot(entry.freq, angle(entry.Z_exp, deg=True), 'or', markersize=4)
        self._freq_axis(entry)
        self.axes.set_xlabel('$f$/Hz')
        self.axes.set_ylabel(r'angle/$^\circ$')

    def Re_data(self, entry):
        if entry.method == 'BHT':
            self.axes.fill_between(entry.freq, entry.mu_Z_H_re_agm-3*entry.band_re_agm, entry.mu_Z_H_re_agm+3*entry.band_re_agm, facecolor='lightgrey')
            self.axes.plot(entry.freq, entry.mu_Z_re, 'k', linewidth=3, label='$Z_\\mu$(Regressed)')
            self.axes.plot(entry.freq, entry.mu_Z_H_re_agm, 'b', linewidth=3, label='$Z_H$(Hilbert transform)')
            self.axes.plot(entry.freq, entry.Z_prime, 'or', markersize=4)
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')
        elif entry.method != 'none':
            self.axes.plot(entry.freq, entry.mu_Z_re, 'k', linewidth=3)
            self.axes.plot(entry.freq, entry.Z_prime, 'or', markersize=4)
        else:
            self.axes.plot(entry.freq, entry.Z_prime, 'or', markersize=4)
        self._freq_axis(entry)
        self.axes.set_xlabel('$f$/Hz')
        self.axes.set_ylabel(r"$Z^{\prime}/(\Omega\,\mathrm{cm}^2)$")

    def Im_data(self, entry):
        if entry.method == 'BHT':
            self.axes.fill_between(entry.freq, -entry.mu_Z_H_im_agm-3*entry.band_im_agm, -entry.mu_Z_H_im_agm+3*entry.band_im_agm, facecolor='lightgrey')
            self.axes.plot(entry.freq, -entry.mu_Z_im, 'k', linewidth=3, label='$Z_\\mu$(Regressed)')
            self.axes.plot(entry.freq, -entry.mu_Z_H_im_agm, 'b', linewidth=3, label='$Z_H$(Hilbert transform)')
            self.axes.plot(entry.freq, -entry.Z_double_prime, 'or', markersize=4)
            self.axes.legend(frameon=False, fontsize=15, loc='upper left')
        elif entry.method != 'none':
            self.axes.plot(entry.freq, -entry.mu_Z_im, 'k', linewidth=3)
            self.axes.plot(entry.freq, -entry.Z_double_prime, 'or', markersize=4)
        else:
            self.axes.plot(entry.freq, -entry.Z_double_prime, 'or', markersize=4)
        self._freq_axis(entry)
        self.axes.set_xlabel('$f$/Hz')
        self.axes.set_ylabel(r"$-Z^{\prime\prime}/(\Omega\,\mathrm{cm}^2)$")

    def Re_residual(self, entry):
        if entry.method == 'none':
            return
        if entry.method == 'BHT':
            self.axes.fill_between(entry.freq, -3*entry.band_re_agm, 3*entry.band_re_agm, facecolor='lightgrey')
            y = entry.res_H_re
            self.axes.plot(entry.freq, y, 'or', markersize=4)
            self.axes.set_ylabel(r"$(R_{\infty}+Z^{\prime}_{H}-Z^{\prime}_{exp})/(\Omega\,\mathrm{cm}^2)$")
            y_max = float(np.max(3*entry.band_re_agm))
        else:
            mod = np.abs(entry.Z_exp)
            y = np.divide(entry.res_re, mod, out=np.zeros_like(entry.res_re, dtype=float), where=mod != 0)*100
            self.axes.plot(entry.freq, y, 'or', markersize=4)
            self.axes.set_ylabel(r"Residual $Z^{\prime}/ \%$")
            y_max = float(np.max(np.abs(y))) if y.size else 0
        self._freq_axis(entry)
        self.axes.set_xlabel('$f$/Hz')
        y_max = max(y_max, np.finfo(float).eps)
        self.axes.set_ylim([-1.1*y_max, 1.1*y_max])

    def Im_residual(self, entry):
        if entry.method == 'none':
            return
        if entry.method == 'BHT':
            self.axes.fill_between(entry.freq, -3*entry.band_im_agm, 3*entry.band_im_agm, facecolor='lightgrey')
            y = entry.res_H_im
            self.axes.plot(entry.freq, y, 'or', markersize=4)
            self.axes.set_ylabel(r"$(\omega L_0+Z^{\prime\prime}_{H}-Z^{\prime\prime}_{exp})/(\Omega\,\mathrm{cm}^2)$")
            y_max = float(np.max(3*entry.band_im_agm))
        else:
            mod = np.abs(entry.Z_exp)
            y = np.divide(entry.res_im, mod, out=np.zeros_like(entry.res_im, dtype=float), where=mod != 0)*100
            self.axes.plot(entry.freq, y, 'or', markersize=4)
            self.axes.set_ylabel(r"Residual $Z^{\prime\prime}/ \%$")
            y_max = float(np.max(np.abs(y))) if y.size else 0
        self._freq_axis(entry)
        self.axes.set_xlabel('$f$/Hz')
        y_max = max(y_max, np.finfo(float).eps)
        self.axes.set_ylim([-1.1*y_max, 1.1*y_max])

    def DRT_data(self, entry, drt_type='Gamma vs Tau'):
        if entry.method == 'none':
            return
        tau = np.asarray(entry.out_tau_vec, dtype=float).reshape(-1)
        omega_like = 1.0/tau
        freq_hz = omega_like/(2*np.pi)
        use_freq = 'Frequency' in drt_type
        use_g = drt_type.startswith('g ')
        x = freq_hz if use_freq else tau
        mult = omega_like if use_g else np.ones_like(tau)

        def line(y, *args, **kwargs):
            self.axes.plot(x, np.asarray(y).reshape(-1)*mult, *args, **kwargs)

        if entry.method == 'simple':
            line(entry.gamma, 'k', linewidth=3); y_min = 0; y_max = np.max(np.asarray(entry.gamma)*mult)
        elif entry.method == 'credit':
            low = np.asarray(entry.lower_bound)*mult; up = np.asarray(entry.upper_bound)*mult
            self.axes.fill_between(x, low, up, facecolor='lightgrey')
            line(entry.mean, 'b', linewidth=3, label='Mean'); line(entry.gamma, 'k', linewidth=3, label='MAP')
            self.axes.legend(frameon=False, fontsize=15); y_min = 0; y_max = np.max(up)
        elif entry.method == 'BHT':
            line(entry.mu_gamma_fine_re, 'b', linewidth=3, label='Mean Re')
            line(entry.mu_gamma_fine_im, 'k', linewidth=3, label='Mean Im')
            vals = np.concatenate([np.asarray(entry.mu_gamma_fine_re)*mult, np.asarray(entry.mu_gamma_fine_im)*mult])
            y_min, y_max = float(np.min(vals)), float(np.max(vals)); self.axes.legend(frameon=False, fontsize=15)
        elif entry.method == 'peak':
            line(entry.gamma, 'k', linewidth=3)
            mat = np.asarray(entry.gamma_gauss_mat)
            for i in range(entry.N_peaks):
                line(mat[:, i], linewidth=3, label=f'Peak {i+1}')
            y_min = 0; y_max = float(np.max(np.asarray(entry.gamma)*mult))
        else:
            return

        self.axes.set_xscale('log')
        self.axes.tick_params(axis='x', which='both', direction='inout')
        if use_freq:
            self.axes.set_xlim(float(np.max(entry.freq_0)), float(np.min(entry.freq_0)))
            self.axes.set_xlabel('$f$/Hz')
            self.axes.set_ylabel(r'$g(f)/(\Omega\,\mathrm{cm}^2/s)$' if use_g else r'$\gamma(\ln f)/(\Omega\,\mathrm{cm}^2)$')
        else:
            lo = 1/(2*np.pi*float(np.max(entry.freq_0)))
            hi = 1/(2*np.pi*float(np.min(entry.freq_0)))
            self.axes.set_xlim(lo, hi)
            self.axes.set_xlabel(r'$\tau$/s')
            self.axes.set_ylabel(r'$g(\tau)/(\Omega\,\mathrm{cm}^2/s)$' if use_g else r'$\gamma(\ln\tau)/(\Omega\,\mathrm{cm}^2)$')
        if np.isfinite(y_max):
            if y_max == y_min:
                y_max = y_min + np.finfo(float).eps
            self.axes.set_ylim(y_min, 1.1*y_max)

    def Score(self, entry):
        """RCs fitting view: input EIS, CNLS fit, and sequential RC contributions."""
        fit = getattr(entry, 'cnls_fit', None)
        self.axes.axhline(0.0, color='lightgray', linewidth=1.0, zorder=0)
        self.axes.plot(
            np.real(entry.Z_exp), -np.imag(entry.Z_exp),
            'or', markersize=5, label='Input EIS'
        )
        if fit:
            Z_fit = np.asarray(fit['Z_fit'], dtype=complex)
            self.axes.plot(
                np.real(Z_fit), -np.imag(Z_fit),
                'k-', linewidth=2.5, label='Bounded-frequency CNLS fit'
            )

            # Plot each fitted parallel-RC contribution sequentially, using the
            # same fitted R/C values shown in the CNLS table. This follows the
            # MATLAB discrete-RC Nyquist display: start at fitted series R,
            # shift only the real part, then use the end of one RC as the
            # horizontal offset for the next RC.
            fit_freq = np.asarray(fit.get('freq', entry.freq), dtype=float).reshape(-1)
            omega = 2.0 * np.pi * fit_freq
            offset = float(fit['R'])
            rc_colors = (
                'tab:blue', 'tab:orange', 'tab:green', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                'tab:cyan', 'navy', 'teal', 'goldenrod'
            )
            for rc_index, rc in enumerate(fit.get('rc', [])):
                R_i = float(rc['R'])
                C_i = float(rc['C'])
                Z_rc = R_i / (1.0 + 1j * omega * R_i * C_i)
                z_real_adj = np.real(Z_rc) + offset
                z_imag = -np.imag(Z_rc)
                self.axes.plot(
                    z_real_adj, z_imag,
                    color=rc_colors[rc_index % len(rc_colors)],
                    linewidth=1.5, label=str(rc['component'])
                )
                offset = float(np.max(z_real_adj))

            self.axes.legend(frameon=False, fontsize=11, loc='best')
        else:
            self.axes.text(
                0.5, 0.06,
                'Run Peak deconvolution first to create the bounded-frequency CNLS fit.',
                transform=self.axes.transAxes, ha='center', va='bottom', fontsize=11
            )
        self.axes.set_xlabel(r"$Z^{\prime}/(\Omega\,\mathrm{cm}^2)$")
        self.axes.set_ylabel(r"$-Z^{\prime\prime}/(\Omega\,\mathrm{cm}^2)$")
        self.axes.axis('equal')



def launch_gui():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = GUI()
    window.show()
    app.exec_()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = GUI(); window.show()
    sys.exit(app.exec_())
