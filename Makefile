PYTHON ?= python3
RUN = OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/muon_revision_matplotlib $(PYTHON)
.PHONY: data check report rank-one controls
data:
	$(PYTHON) download_data.py
check:
	$(RUN) -m pytest -q muon_zero_order_reproduction/tests revision_checks/test_baselines.py rank_one_study/test_rank_one.py
	$(RUN) realdata_ct/audit_walnut.py
	$(RUN) revision_checks/audit_checks.py
	$(RUN) rank_one_study/audit_study.py
	$(RUN) submission_checks/check_mathematics.py
report:
	$(RUN) realdata_ct/report_walnut.py
	$(RUN) revision_checks/report_checks.py
	$(RUN) rank_one_study/plot_results.py
rank-one:
	$(RUN) rank_one_study/run_study.py
	$(RUN) rank_one_study/audit_study.py
	$(RUN) rank_one_study/plot_results.py
controls:
	$(RUN) revision_checks/run_checks.py --phase tune --workers 2
	$(RUN) revision_checks/run_checks.py --phase test --workers 2
	$(RUN) revision_checks/run_checks.py --phase sensitivity --workers 2
	$(RUN) revision_checks/run_checks.py --phase timing --workers 1
	$(RUN) revision_checks/report_checks.py
