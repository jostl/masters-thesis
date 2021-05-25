import re
from collections import defaultdict

import numpy as np
import pandas as pd
from terminaltables import DoubleTable
from pathlib import Path


def weather_table(weather_dict, path):
	table_data = []

	for weather, seeds in weather_dict.items():

		successes = []
		totals = []
		collisions = []
		collided_and_success = []
		total_lights = []
		total_lights_ran = []

		for seed in seeds:
			successes.append(seeds[seed]["success"])
			totals.append(seeds[seed]["total"])
			collisions.append(seeds[seed]["collided"])
			collided_and_success.append(seeds[seed]["collided_and_success"])
			total_lights.append(seeds[seed]["total_lights"])
			total_lights_ran.append(seeds[seed]["total_lights_ran"])

		successes = np.array(successes)
		totals = np.array(totals)
		collisions = np.array(collisions)
		collided_and_success = np.array(collided_and_success)
		total_lights = np.array(total_lights)
		total_lights_ran = np.array(total_lights_ran)

		success_rates = successes / totals * 100
		lights_ran_rates = total_lights_ran / total_lights * 100
		timeouts = totals - successes - collisions + collided_and_success

		collision_rates = collisions / totals * 100
		timeout_rates = timeouts / totals * 100

		collided_and_success_rates= collided_and_success / totals * 100

		for elem in abs(timeout_rates + collision_rates + success_rates - collided_and_success_rates):
			assert 99.9 < elem < 100.1, "rates do not sum to 100"

		if len(seeds) > 1:
			table_data.append([weather, "%.1f ± %.1f" % (np.mean(success_rates), np.std(success_rates, ddof=1)),
							   "%d/%d" % (sum(successes), sum(totals)), ','.join(sorted(seeds.keys())),
							   "%.1f ± %.1f" % (np.mean(collision_rates), np.std(collision_rates, ddof=1)),
							   "%.1f ± %.1f" % (np.mean(timeout_rates), np.std(timeout_rates, ddof=1)),
							   "%.1f ± %.1f" % (np.mean(lights_ran_rates), np.std(lights_ran_rates, ddof=1)),
							   "%d" % np.sum(collided_and_success)])
		else:
			table_data.append([weather, "%.1f" % np.mean(success_rates), "%d/%d" % (sum(successes), sum(totals)),
							   ','.join(sorted(seeds.keys())),
							   "%.1f" % collision_rates, "%.1f" % timeout_rates, "%.1f" % lights_ran_rates,
							   "%.d" % collided_and_success])

	table_data = sorted(table_data, key=lambda row: row[0])
	table_data = [('Weather', 'Success Rate %', 'Total', 'Seeds', "Collision %", "Timeout %", "Lights ran %",
				   "Collided+Success")] + table_data
	table = DoubleTable(table_data, "Performance of %s" % path.name)
	print(table.table)


def main(path_name, separate_seeds=False, create_weather_table=False):

	performance = dict()
	weather_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

	path = Path(path_name)
	for summary_path in path.glob('*/summary.csv'):
		name = summary_path.parent.name
		match = re.search('^(?P<suite_name>.*Town.*-v[0-9]+.*)_seed(?P<seed>[0-9]+)', name)
		suite_name = match.group('suite_name')
		seed = match.group('seed')

		summary = pd.read_csv(summary_path)

		if suite_name not in performance:
			performance[suite_name] = dict()

		collided_and_success_dataframe = np.logical_and(summary["success"], summary["collided"])

		performance[suite_name][seed] = (summary['success'].sum(), len(summary), summary["collided"].sum(),
										 collided_and_success_dataframe.sum(),
										 summary["total_lights"].sum(), summary["total_lights_ran"].sum())

		if create_weather_table:

			# need to iterate over each route
			for i in range(len(summary)):
				weather_dict[summary["weather"][i]][seed]["success"] += summary["success"][i]
				weather_dict[summary["weather"][i]][seed]["total"] += 1
				weather_dict[summary["weather"][i]][seed]["collided"] += summary["collided"][i]
				weather_dict[summary["weather"][i]][seed]["collided_and_success"] += np.logical_and(summary["success"][i], summary["collided"][i])
				weather_dict[summary["weather"][i]][seed]["total_lights"] += summary["total_lights"][i]
				weather_dict[summary["weather"][i]][seed]["total_lights_ran"] += summary["total_lights_ran"][i]

	if create_weather_table:
		weather_table(weather_dict, path)
		return



	table_data = []
	for suite_name, seeds in performance.items():

		if separate_seeds:
			for seed in seeds:
				successes, totals, collisions, collided_and_success, total_lights, total_lights_ran = np.array(seeds[seed])
				success_rates = successes / totals * 100
				lights_ran_rates = total_lights_ran / total_lights * 100
				timeouts = totals - successes - collisions + collided_and_success

				collision_rates = collisions / totals * 100
				timeout_rates = timeouts / totals * 100

				table_data.append(
					[suite_name+"-seed-"+seed, "%.1f" % success_rates, "%d/%d" % (successes, totals),
					 ','.join(seed),
					 "%.1f" % collision_rates, "%.1f" % timeout_rates, "%.1f" % lights_ran_rates,
					 "%d" % collided_and_success])


		else:

			successes, totals, collisions, collided_and_success, total_lights, total_lights_ran = np.array(list(zip(*seeds.values())))
			success_rates = successes / totals * 100
			lights_ran_rates = total_lights_ran / total_lights * 100
			timeouts = totals - successes - collisions + collided_and_success

			collision_rates = collisions / totals * 100
			timeout_rates = timeouts / totals * 100

			if len(seeds) > 1:
				table_data.append([suite_name, "%.1f ± %.1f"%(np.mean(success_rates), np.std(success_rates, ddof=1)),
								   "%d/%d"%(sum(successes),sum(totals)), ','.join(sorted(seeds.keys())),
								   "%.1f ± %.1f"%(np.mean(collision_rates), np.std(collision_rates, ddof=1)),
								   "%.1f ± %.1f"%(np.mean(timeout_rates), np.std(timeout_rates, ddof=1)),
								   "%.1f ± %.1f"%(np.mean(lights_ran_rates), np.std(lights_ran_rates, ddof=1)),
								   "%d"%np.sum(collided_and_success)])
			else:
				table_data.append([suite_name, "%.1f"%np.mean(success_rates), "%d/%d"%(sum(successes),sum(totals)), ','.join(sorted(seeds.keys())),
									"%.1f"%collision_rates, "%.1f"%timeout_rates, "%.1f"%lights_ran_rates, "%d"%collided_and_success])

	table_data = sorted(table_data, key=lambda row: row[0])
	table_data = [('Suite Name', 'Success Rate %', 'Total', 'Seeds', "Collision %", "Timeout %", "Lights ran %", "Collided+Success")] + table_data
	table = DoubleTable(table_data, "Performance of %s"%path.name)
	print(table.table)



if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--path', help='path of benchmark folder')
	parser.add_argument("--separate-seeds", action="store_true")
	parser.add_argument("--weather", action="store_true")

	args = parser.parse_args()
	main(args.path, args.separate_seeds, create_weather_table=args.weather)
