"""
Walk through the whole GUI the way a user would, without a display: every
template, every figure, every menu action, every layer type and material
class, contacts, settings, save / open, export, stop, and bad input. Any
uncaught exception, any figure that fails to draw and any unexpected solver
error is collected and reported.

    python tests/test_gui_walkthrough.py          # prints each step
    pytest tests/test_gui_walkthrough.py

Dialogs are replaced by stubs, so nothing blocks and no browser opens.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402


def walkthrough(verbose: bool = False) -> list:
    from gui import theme, templates
    from gui.app import App, _FIGURES
    from gui.layer_editor import _TYPE_NAMES
    from devices.layer import AbruptLayer, GradedLayer
    from physics.materials.alloys import KINDS
    from physics.rsm import reflections_for

    problems: list = []
    step = [""]

    def log(text):
        step[0] = text
        if verbose:
            print(text, flush=True)

    def bad(text):
        problems.append(f"[{step[0]}] {text}")
        if verbose:
            print("   PROBLEM:", text, flush=True)

    def hook(etype, value, tb):
        bad("uncaught exception: " + "".join(traceback.format_exception(etype, value, tb))[-900:])
    sys.excepthook = hook

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("EpiBand")
    theme.apply(app)
    tmp = tempfile.mkdtemp(prefix="epiband_gui_")

    # ---- stubs for everything modal or external ----
    dialog = {"save": "", "open": ""}
    opened_urls, boxes = [], []
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (dialog["save"], ""))
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (dialog["open"], ""))
    QtWidgets.QMessageBox.exec = lambda self: boxes.append(self.text()) or 0
    QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: boxes.append("critical: " + str(a[-1])) or 0)
    QtGui.QDesktopServices.openUrl = staticmethod(lambda url: opened_urls.append(url.toString()) or True)

    w = App()
    w.resize(1360, 860)
    w.show()

    def pump(seconds=0.15):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def wait_solve(timeout=300):
        t0 = time.time()
        pump(0.2)
        while w._busy and time.time() - t0 < timeout:
            pump(0.1)
        pump(0.6)                       # debounce timers
        if w._busy:
            bad("solve still running after timeout")
        return w._status_message.text()

    def run(expect_ok=True):
        w.act_run.trigger()
        status = wait_solve()
        if expect_ok and not status.startswith("Converged"):
            bad(f"run did not converge cleanly: {status!r}")
        return status

    def figure_text(name):
        fig = w.plot_panel._tabs[name][0]
        return " ".join(t.get_text() for ax in fig.axes for t in ax.texts)

    def check_figures(label, expect_data=True):
        for name, _short, _long in _FIGURES:
            w.plot_panel.show_tab(name)
            pump(0.12)
            text = figure_text(name)
            if "Could not render" in text:
                bad(f"{label}: figure {name!r} failed: {text[:300]}")
            fig = w.plot_panel._tabs[name][0]
            # Recombination shows a note instead of curves at equilibrium and in the
            # flat-quasi-Fermi-level mode; its curves are checked under bias below
            if expect_data and name not in ("RSM", "LB", "QCSE", "Wavefunctions", "Recombination") and not any(
                    ax.lines or ax.collections or ax.images for ax in fig.axes):
                bad(f"{label}: figure {name!r} is empty")
        w.plot_panel.show_tab("Band Diagram")

    # ------------------------------------------------------------------
    log("empty project")
    w.act_run.trigger()
    pump()
    if "Add at least one layer" not in w._status_message.text():
        bad(f"Run on an empty project: {w._status_message.text()!r}")
    check_figures("empty", expect_data=False)
    w.plot_panel._calculate_rsm()
    w._solve_light_bands()
    w.act_delete.trigger()
    w.act_stop.trigger()
    pump()

    # ------------------------------------------------------------------
    for tpl in templates.TEMPLATES:
        log(f"template: {tpl.title}")
        w._load_template(tpl.key)
        status = wait_solve()
        if not status.startswith("Converged"):
            bad(f"template did not converge: {status!r}")
            continue
        if tpl.title not in w.windowTitle():
            bad(f"window title {w.windowTitle()!r}")
        result = w.plot_panel.results[0]
        if not w.plot_panel.summary_widget.toPlainText().startswith("SOLUTION"):
            bad("summary pane is empty")
        if abs(result.V_applied) < 1e-12 and (result.R_srh.any() or result.R_rad.any() or result.R_aug.any()):
            bad("non-zero recombination at equilibrium")
        check_figures(tpl.key)

        # every region of the Region dropdown, on every position figure
        combo = w.plot_panel._zoom_combo
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            for name in ("Band Diagram", "Carriers", "Recombination", "Fields", "Strain"):
                w.plot_panel.show_tab(name)
                pump(0.03)
                if "Could not render" in figure_text(name):
                    bad(f"region {combo.itemText(i)!r}, figure {name!r}: {figure_text(name)[:200]}")
        combo.setCurrentIndex(0)

        # style checkboxes
        w.plot_panel.show_tab("Band Diagram")
        for box in w.plot_panel.style_panel.findChildren(QtWidgets.QCheckBox):
            box.toggle()
            pump(0.03)
            if "Could not render" in figure_text("Band Diagram"):
                bad(f"style option {box.text()!r} breaks the band diagram")
            box.toggle()

        # reciprocal space map, every reflection of this crystal
        w.plot_panel.show_tab("Strain")
        rsm_combo = w.plot_panel._rsm_combo
        labels = [rsm_combo.itemText(i) for i in range(rsm_combo.count())]
        if labels != list(reflections_for(result.crystal)):
            bad(f"reflections offered {labels} do not fit a {result.crystal} structure")
        for label in labels:
            rsm_combo.setCurrentText(label)
            w.plot_panel._calculate_rsm()
            pump(0.05)
            if w.plot_panel._rsm is None or "Could not render" in figure_text("RSM"):
                bad(f"RSM {label}: {w._status_message.text()!r} {figure_text('RSM')[:200]}")
            elif "RECIPROCAL SPACE MAP" not in w.plot_panel.summary_widget.toPlainText():
                bad(f"RSM {label}: not in the summary")

        # export the figure on screen in the three formats
        for ext in ("png", "pdf", "svg"):
            dialog["save"] = os.path.join(tmp, f"{tpl.key}.{ext}")
            w.act_export.trigger()
            pump(0.05)
            if not os.path.isfile(dialog["save"]) or os.path.getsize(dialog["save"]) < 500:
                bad(f"export to {ext} wrote nothing")

        # save, reopen, and the reopened project must solve to the same result
        dialog["save"] = os.path.join(tmp, f"{tpl.key}_project")          # no extension on purpose
        w.act_save_as.trigger()
        path = dialog["save"] + ".json"
        if not os.path.isfile(path):
            bad("Save As wrote no file")
            continue
        w.act_save.trigger()
        ns_before = float(result.n.sum())
        w.act_new.trigger()
        pump()
        if w.model.layers or "Untitled" not in w.windowTitle():
            bad("New did not clear the project")
        dialog["open"] = path
        w.act_open.trigger()
        pump()
        if os.path.basename(path) not in w.windowTitle():
            bad(f"Open: title {w.windowTitle()!r}")
        status = run()
        if w.plot_panel.results and abs(float(w.plot_panel.results[0].n.sum()) - ns_before) > 1e-6 * abs(ns_before):
            bad("the reopened project gives a different solution")

    # bands under illumination, one structure of each family
    for key, fields in (("pn", {}), ("gaas_led", {"T_growth_C": "27", "wavelength_nm": "600"})):
        log(f"bands under illumination: {key}")
        w._load_template(key)
        wait_solve()
        for name, value in fields.items():
            w._growth_fields[name].setText(value)
        w._solve_light_bands()
        pump(0.3)
        status = w._status_message.text()
        if not status.startswith("Bands under light: photovoltage") or "NOT CONVERGED" in status:
            bad(f"illumination: {status!r}")
        w.plot_panel.show_tab("LB")
        pump(0.2)
        fig = w.plot_panel._tabs["LB"][0]
        if "Could not render" in figure_text("LB") or not any(ax.lines for ax in fig.axes):
            bad(f"illuminated-bands figure: {figure_text('LB')[:200]}")
    w._growth_fields["power_W_cm2"].setText("abc")
    w._solve_light_bands()
    if "failed" not in w._status_message.text():
        bad("non-numeric illumination input was not reported")

    # ------------------------------------------------------------------
    log("menus, view toggles, help")
    w._load_template("gaas_hemt")
    wait_solve()
    for i, act in enumerate(w._figure_actions):
        act.trigger()
        pump(0.05)
        if w.plot_panel.current_tab() != i:
            bad(f"View menu entry {act.text()!r} shows tab {w.plot_panel.current_tab()}")
    for act in (w.act_side, w.act_output):
        act.trigger()
        pump(0.05)
        act.trigger()
    if not w.side_tabs.isVisible():
        bad("side panel did not come back")
    for i in range(w.side_tabs.count()):
        w.side_tabs.setCurrentIndex(i)
        pump(0.05)
    w.side_tabs.setCurrentIndex(0)
    w.act_about.trigger()
    if not boxes or "EpiBand" not in boxes[-1]:
        bad("About box")
    w.act_docs.trigger()
    w.act_results.trigger()
    if len(opened_urls) != 2:
        bad(f"Manual / Results folder opened {opened_urls}")
    for mode in ("pan", "zoom", "pan"):
        w.plot_panel._set_mode(mode)
    w.plot_panel._home_btn.click()

    # ------------------------------------------------------------------
    for family, start, first_material in (("nitride", "hemt", "AlGaN"), ("zincblende", "gaas_hemt", "AlGaAs")):
        log(f"layer editing: {family}")
        w._load_template(start)
        wait_solve()
        n0 = len(w.model.layers)
        sp, ed = w.stack_panel, w.editor_panel
        for method in ("_add_abrupt", "_add_graded", "_add_quantum_marker", "_add_surface_charge",
                       "_add_interface_dipole"):
            getattr(sp, method)()
            pump(0.05)
        if len(w.model.layers) != n0 + 5:
            bad("adding the five entry types")
        new = w.model.layers[n0]
        if getattr(new, "material", None) != first_material:
            bad(f"a new layer in a {family} stack is {getattr(new, 'material', None)}")

        # switch the selected entry through every type and back
        sp.select(n0)
        for kind in _TYPE_NAMES + ["Abrupt", "Graded", "Abrupt"]:
            ed._switch_type(kind)
            pump(0.05)
        if not isinstance(w.model.layers[n0], AbruptLayer):
            bad("type switching did not end on an abrupt layer")

        # every material class of this family, abrupt and graded
        kinds = [k for k, v in KINDS.items() if (v[0] == "zincblende") == (family == "zincblende")]
        for graded in (False, True):
            ed._switch_type("Graded" if graded else "Abrupt")
            for kind in kinds:
                combo = ed._vars["Material"][0]
                combo.setCurrentText(kind)
                pump(0.05)
                layer = w.model.layers[n0]
                if layer.material != kind:
                    bad(f"material {kind} ({'graded' if graded else 'abrupt'}) was not applied: {layer.material}")
                    continue
                for label, (widget, vkind) in list(ed._vars.items()):
                    if "fraction" in label or " start " in label or " end " in label:
                        widget.setText("0.2" if "end" not in label else "0.3")
                ed._apply()
                pump(0.03)
                title = sp._table.item(len(w.model.layers) - 1 - n0, 1).text()
                if not title or title == "Layer":
                    bad(f"layer table label for {kind}: {title!r}")
        ed._switch_type("Abrupt")
        ed._vars["Material"][0].setCurrentText(first_material)

        # bad and edge input must not crash or corrupt the layer
        before = w.model.layers[n0]
        for label, text in (("Thickness (nm)", "abc"), ("Thickness (nm)", ""), ("n-doping (cm^-3)", "1e"),
                            ("Al fraction (0-1)", "1.7"), ("Al fraction (0-1)", "-0.2")):
            ed._vars[label][0].setText(text)
            ed._apply()
        if w.model.layers[n0] != before:
            bad("invalid input changed the layer")
        ed.show(n0)
        for label, text in (("Thickness (nm)", "12.5"), ("n-doping (cm^-3)", "2e17"), ("Al fraction (0-1)", "0.25"),
                            ("Grid spacing (nm, blank=default)", "0.5"),
                            ("Custom strain εxx (blank=auto, -=compressive, +=tensile)", "-0.002")):
            ed._vars[label][0].setText(text)
            ed._vars[label][0].textEdited.emit(text)
        ed.flush_pending()
        layer = w.model.layers[n0]
        if (layer.thickness_nm, layer.n_doping, layer.x_Al, layer.dx_nm, layer.custom_strain_xx) != (
                12.5, 2e17, 0.25, 0.5, -0.002):
            bad(f"field edits not applied: {layer}")
        ed._vars["Strain-relaxed"][0].setChecked(True)
        if not w.model.layers[n0].relaxed:
            bad("Strain-relaxed checkbox")

        # surface states: add and remove levels; dipole and marker fields
        sp.select(n0 + 3)
        ed._add_surface_state()
        ed._add_surface_state()
        ed._remove_surface_state(0)
        if len(w.model.layers[n0 + 3].states) != 2:
            bad("surface-state levels")
        ed._surface_rows[0][0].setText("3e12")
        ed._surface_rows[0][2].setCurrentText("acceptor")
        ed._apply()
        sp.select(n0 + 4)
        ed._vars["Sheet charge (C/m^2)"][0].setText("2e-3")
        ed._apply()
        sp.select(n0 + 2)
        ed._vars["Boundary"][0].setCurrentText("end")

        # move up / down / delete through the buttons and the table
        sp.select(n0)
        sp._down_btn.click()
        sp._up_btn.click()
        sp._up_btn.click()
        if sp.selected_index != n0 + 1:
            bad(f"Up / Down moved the selection to {sp.selected_index}")
        sp._table.selectRow(0)
        pump(0.05)
        if sp.selected_index != len(w.model.layers) - 1:
            bad("clicking a table row does not select it")
        for _ in range(4):                      # remove the graded layer and the three interface entries
            sp.select(len(w.model.layers) - 1)
            w.act_delete.trigger()
        sp.select(None)
        w.act_delete.trigger()                  # nothing selected: no effect
        if len(w.model.layers) != n0 + 1 or not isinstance(w.model.layers[-1], (AbruptLayer, GradedLayer)):
            bad(f"delete left {len(w.model.layers)} entries")
        status = run(expect_ok=False)
        if not (status.startswith("Converged") or status.startswith("Did not converge")):
            bad(f"edited {family} structure: {status!r}")
        check_figures(f"edited {family}")

    log("mixed crystal families")
    w._load_template("gaas_hemt")
    wait_solve()
    w.stack_panel._add_abrupt()
    w.editor_panel._vars["Material"][0].setCurrentText("AlGaN")
    status = run(expect_ok=False)
    if "one crystal system" not in status:
        bad(f"mixing nitride and arsenide layers gives {status!r}")
    w.stack_panel.delete_selected()
    run()

    # ------------------------------------------------------------------
    log("contacts")
    w._load_template("gaas_hemt")
    wait_solve()
    cp = w.contacts_panel
    for combo in cp.findChildren(QtWidgets.QComboBox):
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            pump(0.01)
    for spin in cp.findChildren(QtWidgets.QDoubleSpinBox):
        spin.setValue(0.75)
    types = cp.findChildren(QtWidgets.QComboBox)
    types[0].setCurrentIndex(types[0].findData("schottky"))
    types[2].setCurrentIndex(types[2].findData("ohmic"))
    if (w.model.top_contact.contact_type, w.model.top_contact.barrier_eV, w.model.bottom_contact.contact_type) != (
            "schottky", 0.75, "ohmic"):
        bad(f"contacts not applied: {w.model.top_contact} {w.model.bottom_contact}")
    run()

    # ------------------------------------------------------------------
    log("simulation settings")
    w._load_template("gaas_led")
    wait_solve()
    st = w.settings_panel
    for attr, text, expect in (("T", "250", 250.0), ("dx_nm", "2", 2.0), ("V_applied", "1.1", 1.1),
                               ("max_iter", "300", 300), ("tol", "1e-5", 1e-5), ("alpha", "0.2", 0.2),
                               ("n_states_e", "8", 8), ("n_states_h", "8", 8),
                               ("tau_n_ns", "5", 5.0), ("tau_p_ns", "5", 5.0), ("B_rad", "2e-10", 2e-10),
                               ("C_auger", "1e-29", 1e-29)):
        edit = st._field_vars[attr][0]
        edit.setText(text)
        edit.textEdited.emit(text)
    st.flush_pending()
    for attr, _t, expect in (("T", 0, 250.0), ("V_applied", 0, 1.1), ("tau_n_ns", 0, 5.0), ("C_auger", 0, 1e-29),
                             ("n_states_e", 0, 8)):
        if getattr(w.model.settings, attr) != expect:
            bad(f"setting {attr} = {getattr(w.model.settings, attr)!r}, expected {expect!r}")
    status = run()
    r = w.plot_panel.results[0]
    if (r.T, r.V_applied) != (250.0, 1.1):
        bad(f"solve used T = {r.T}, V = {r.V_applied}")
    if "RECOMBINATION" not in w.plot_panel.summary_widget.toPlainText():
        bad("no recombination block in the summary of a biased solve")
    w.plot_panel.show_tab("Recombination")
    pump(0.1)
    rec_ax = w.plot_panel._tabs["Recombination"][0].axes[0]
    if len(rec_ax.lines) < 4 or "equilibrium" in figure_text("Recombination"):
        bad("the Recombination figure has no curves under bias")
    j_fast = abs(r.J_total)
    for attr in ("tau_n_ns", "tau_p_ns", "B_rad", "C_auger"):
        edit = st._field_vars[attr][0]
        edit.setText("")
        edit.textEdited.emit("")
    st.flush_pending()
    if w.model.settings.recombination() is not None:
        bad("clearing the recombination fields did not restore the material values")
    for text in ("abc", "-3", "0"):                 # rejected, keep the previous value
        edit = st._field_vars["tau_n_ns"][0]
        edit.setText(text)
        edit.textEdited.emit(text)
        st.flush_pending()
    if w.model.settings.tau_n_ns is not None:
        bad(f"invalid lifetime accepted: {w.model.settings.tau_n_ns!r}")
    st._field_vars["tau_n_ns"][0].setText("")
    run()
    if abs(abs(w.plot_panel.results[0].J_total) - j_fast) < 1e-9 * j_fast:
        bad("recombination coefficients have no effect on the current")
    st._adv_cb.toggle()
    st._adv_cb.toggle()

    log("model switches")
    w._load_template("stark")
    wait_solve()
    st = w.settings_panel
    for box in st.findChildren(QtWidgets.QCheckBox):
        box.toggle()
        status = run(expect_ok=False)
        if not status.startswith("Converged"):
            bad(f"with {box.text()!r} toggled: {status!r}")
        check_figures(f"toggle {box.text()}")
        box.toggle()
    for combo in st.findChildren(QtWidgets.QComboBox):
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            status = run(expect_ok=False)
            if not status.startswith("Converged"):
                bad(f"with {combo.itemText(i)!r}: {status!r}")
        combo.setCurrentIndex(0)

    log("bias with flat quasi-Fermi levels")
    w._load_template("gaas_led")
    wait_solve()
    w.model.settings.flat_qfl = True
    w.model.settings.V_applied = 0.8
    run()
    check_figures("flat qfl")

    # ------------------------------------------------------------------
    log("stop a running solve")
    w._load_template("uvled")
    wait_solve()
    w.model.settings.V_applied = 6.0                 # a long solve: there is time to stop it
    w.act_run.trigger()
    pump(0.3)
    if not w._busy:
        bad("the solve was not running")
    if w.act_run.isEnabled() or not w.act_stop.isEnabled():
        bad("Run / Stop enabled state while solving")
    w.act_run.trigger()                              # ignored while busy
    w.act_stop.trigger()
    status = wait_solve(120)
    if not status.startswith("Stopped"):
        bad(f"after Stop: {status!r}")
    if not w.act_run.isEnabled() or w.act_stop.isEnabled():
        bad("Run / Stop enabled state after stopping")
    w._load_template("gaas_hemt")
    if not wait_solve().startswith("Converged"):
        bad("cannot solve after a stopped run")

    log("broken project file")
    path = os.path.join(tmp, "broken.json")
    open(path, "w").write("{ not json")
    dialog["open"] = path
    n_boxes = len(boxes)
    w.act_open.trigger()
    if len(boxes) != n_boxes + 1 or "Could not load" not in boxes[-1]:
        bad("a broken project file is not reported")
    dialog["open"] = dialog["save"] = ""             # cancelled dialogs
    for act in (w.act_open, w.act_save_as, w.act_export):
        act.trigger()

    log("close")
    w.close()
    pump(0.2)
    return problems


def test_gui_walkthrough():
    problems = walkthrough()
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    found = walkthrough(verbose=True)
    print(f"\n{len(found)} problem(s)")
    for p in found:
        print(" -", p)
    sys.exit(1 if found else 0)
