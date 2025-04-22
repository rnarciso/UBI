import streamlit as st
import pandas as pd
import numpy as np
import datetime # Para obter o ano atual como sugestão

# --- Default Parameters (Ajustados e com Fertilidade) ---
# Valores internos representam a escala usada nos cálculos (ex: 0.04 para 4%)
DEFAULT_PARAMS = {
    # Parâmetros da RBU
    "ubi_annual_amount_eur": 14400.0, # Usar float para consistência
    "ubi_start_year": 2027,           # Inteiro
    "ubi_eligible_population_share": 1.0,

    # Parâmetros Econômicos e Comportamentais
    "labor_participation_reduction_factor": 0.04,
    "labor_hours_reduction_factor": 0.01,
    "productivity_growth_boost_pp": 0.001,
    "consumption_propensity_ubi_recipients": 0.8,
    "consumption_reduction_high_income_financing": 0.2,

    # Parâmetros Sociais
    "wellbeing_satisfaction_boost": 0.5,
    "mental_health_reduction_pct": 40.0,
    "crime_reduction_per_gini_point": 0.05,
    "unpaid_work_increase_factor": 1.1,

    # Parâmetros de Financiamento e Reação
    "financing_model": "mixed_taxes_debt",
    "additional_tax_rate_for_ubi": 0.05,
    "capital_flight_factor_high_tax": 0.005,
    "migration_net_increase_factor_restricted": 1.25,

    # Parâmetros de Inflação
    "baseline_annual_inflation_rate": 0.02,
    "inflation_ubi_demand_sensitivity": 0.15,
    "inflation_productivity_dampening_factor": 0.5,

    # Impacto Populacional Adicional
    "fertility_rate_ubi_boost_pp": 0.001,

    # Parâmetros de Simulação - Default para inputs
    "start_year": datetime.date.today().year, # Começa no ano atual por padrão
    "end_year": datetime.date.today().year + 50, # Simula 50 anos por padrão

    # Dados Iniciais
    "initial_population_millions": 84.0,
    "initial_gdp_trillion_eur": 4.0,
    "initial_participation_rate": 0.75,
    "initial_avg_hours_worked_per_worker": 1600.0, # Float
    "initial_unemployment_rate": 0.055,
    "initial_gini_coefficient": 0.29,
    "initial_poverty_rate": 0.15,
    "initial_wellbeing_satisfaction": 7.0,
    "initial_mental_health_prevalence": 0.10,
    "initial_crime_rate_index": 100.0,
    "initial_net_migration_thousands": 100.0,
    "initial_millionaires_thousands": 150.0,
    "initial_govt_debt_gdp_ratio": 0.60,
    "baseline_annual_pop_growth_rate": 0.001,
    "baseline_annual_gdp_growth_rate": 0.012,
    "baseline_annual_productivity_growth_rate": 0.01,
    "poverty_floor": 0.03
}


# --- Dicionário de Formatação ---
formatters_dict = {
    # ... (igual ao anterior, sem mudanças)
    "Population (Millions)": "{:.1f}",
    "UBI Cost (Billion EUR)": "{:.1f}",
    "Labor Participation Rate": "{:.1%}",
    "Avg Hours Worked": "{:.0f}",
    "Unemployment Rate": "{:.1%}",
    "Labor Force (Millions)": "{:.1f}",
    "GDP Real (Trillion EUR)": "{:.2f}",
    "GDP Growth Rate": "{:.2%}",
    "Productivity Growth Rate": "{:.2%}",
    "Productivity Level (Index)": "{:.1f}",
    "Consumption (Trillion EUR)": "{:.2f}",
    "Investment (Trillion EUR)": "{:.2f}",
    "Govt Debt/GDP Ratio": "{:.1%}",
    "Additional Tax Revenue (Billion EUR)": "{:.1f}",
    "Capital Flight Index": "{:.1f}",
    "Gini Coefficient": "{:.3f}",
    "Poverty Rate": "{:.1%}",
    "Wellbeing Satisfaction (0-10)": "{:.1f}",
    "Mental Health Prevalence": "{:.2%}",
    "Crime Rate Index": "{:.1f}",
    "Unpaid Work Index": "{:.1f}",
    "Net Migration (Thousands)": "{:.0f}",
    "Millionaires (Thousands)": "{:.0f}",
    "Inflation Rate": "{:.2%}"
}

# --- Classe UBISimulator (sem alterações na lógica interna) ---
class UBISimulator:
    # ... (código da classe exatamente como na resposta anterior) ...
    # ... (_initialize_scenario, run_simulation, get_results, get_summary_dataframe) ...
    """
    Simulador de Impactos de Longo Prazo da Renda Básica Universal (RBU).
    Modela cenários 'Com RBU' vs. 'Sem RBU' (Baseline) ao longo do tempo.
    """
    def __init__(self, params=None):
        if params is None:
             params = DEFAULT_PARAMS.copy()
        # Garante tipos numéricos corretos ao receber params
        self.params = {}
        for k, v in params.items():
             # Heurística simples para tipos
             if isinstance(v, bool): # Preserva booleanos se houver
                 self.params[k] = v
             elif k in ['start_year', 'end_year', 'ubi_start_year']:
                 try: self.params[k] = int(v)
                 except (ValueError, TypeError): self.params[k] = int(DEFAULT_PARAMS[k]) # Fallback
             elif k == 'financing_model':
                 self.params[k] = str(v)
             else: # Tenta converter outros para float
                 try: self.params[k] = float(v)
                 except (ValueError, TypeError): self.params[k] = v # Mantem original se falhar

        # Garante que end_year seja pelo menos start_year (como int)
        self.params["end_year"] = max(int(self.params["start_year"]), int(self.params["end_year"]))
        self.years = np.arange(int(self.params["start_year"]), int(self.params["end_year"]) + 1)
        self.results = {}

    def _initialize_scenario(self):
        """Cria um DataFrame para armazenar os resultados anuais de um cenário."""
        indicators = list(formatters_dict.keys()) # Usa as chaves do formatador como lista de indicadores
        index_years = self.years if len(self.years) > 0 else [int(self.params["start_year"])]
        df = pd.DataFrame(index=index_years, columns=indicators, dtype=float)

        # Preenche valores iniciais para o ano base (start_year - 1) para cálculos
        base_year = int(self.params["start_year"]) - 1
        # Garante que o índice base existe antes de tentar preencher
        if base_year not in df.index:
             # Adiciona linha NaN se não existir, garantindo que seja float
             df.loc[base_year] = np.full(len(indicators), np.nan, dtype=float)


        df.loc[base_year, "Population (Millions)"] = float(self.params["initial_population_millions"])
        df.loc[base_year, "GDP Real (Trillion EUR)"] = float(self.params["initial_gdp_trillion_eur"])
        df.loc[base_year, "Labor Participation Rate"] = float(self.params["initial_participation_rate"])
        df.loc[base_year, "Avg Hours Worked"] = float(self.params["initial_avg_hours_worked_per_worker"])
        df.loc[base_year, "Unemployment Rate"] = float(self.params["initial_unemployment_rate"])
        df.loc[base_year, "Productivity Level (Index)"] = 100.0 # Base 100
        df.loc[base_year, "Gini Coefficient"] = float(self.params["initial_gini_coefficient"])
        df.loc[base_year, "Poverty Rate"] = float(self.params["initial_poverty_rate"])
        df.loc[base_year, "Wellbeing Satisfaction (0-10)"] = float(self.params["initial_wellbeing_satisfaction"])
        df.loc[base_year, "Mental Health Prevalence"] = float(self.params["initial_mental_health_prevalence"])
        df.loc[base_year, "Crime Rate Index"] = float(self.params["initial_crime_rate_index"])
        df.loc[base_year, "Net Migration (Thousands)"] = float(self.params["initial_net_migration_thousands"])
        df.loc[base_year, "Millionaires (Thousands)"] = float(self.params["initial_millionaires_thousands"])
        df.loc[base_year, "Govt Debt/GDP Ratio"] = float(self.params["initial_govt_debt_gdp_ratio"])
        df.loc[base_year, "Capital Flight Index"] = 100.0
        df.loc[base_year, "Unpaid Work Index"] = 100.0
        df.loc[base_year, "Inflation Rate"] = float(self.params["baseline_annual_inflation_rate"])

        # Valores derivados iniciais
        pop_base = df.loc[base_year, "Population (Millions)"]
        part_rate_base = df.loc[base_year, "Labor Participation Rate"]
        unemp_rate_base = df.loc[base_year, "Unemployment Rate"]
        if pd.notna(pop_base) and pop_base > 0 and pd.notna(part_rate_base) and pd.notna(unemp_rate_base):
            initial_labor_force = pop_base * part_rate_base
            df.loc[base_year, "Labor Force (Millions)"] = initial_labor_force * (1.0 - unemp_rate_base) # Empregados
        else:
            df.loc[base_year, "Labor Force (Millions)"] = 0.0

        gdp_base = df.loc[base_year, "GDP Real (Trillion EUR)"]
        if pd.notna(gdp_base) and gdp_base > 0:
            df.loc[base_year, "Consumption (Trillion EUR)"] = gdp_base * 0.6
            df.loc[base_year, "Investment (Trillion EUR)"] = gdp_base * 0.2
        else:
            df.loc[base_year, "Consumption (Trillion EUR)"] = 0.0
            df.loc[base_year, "Investment (Trillion EUR)"] = 0.0

        # Inicializa outras colunas que podem não ter valor base explícito
        df.loc[base_year, "UBI Cost (Billion EUR)"] = 0.0
        df.loc[base_year, "GDP Growth Rate"] = 0.0
        # Garante que Productivity Growth Rate tenha valor inicial
        if pd.isna(df.loc[base_year, "Productivity Growth Rate"]):
             df.loc[base_year, "Productivity Growth Rate"] = float(self.params["baseline_annual_productivity_growth_rate"])
        df.loc[base_year, "Additional Tax Revenue (Billion EUR)"] = 0.0

        return df

    def run_simulation(self):
        """Executa a simulação para os cenários Baseline e Com RBU."""
        start_year_int = int(self.params["start_year"])
        base_year = start_year_int - 1

        if len(self.years) == 0:
             # Handle case where start_year == end_year
             initialized_df = self._initialize_scenario()
             # Check if base_year exists, otherwise use start_year_int
             year_to_use = base_year if base_year in initialized_df.index else start_year_int
             if year_to_use not in initialized_df.index: # Failsafe if neither exists
                  st.error(f"Cannot initialize simulation for year {start_year_int}")
                  return

             single_year_data = initialized_df.loc[[year_to_use]].rename(index={year_to_use: start_year_int})
             baseline_df = single_year_data.fillna(0.0)
             ubi_df = single_year_data.fillna(0.0) # UBI won't be active anyway
             self.results["Baseline (Sem RBU)"] = baseline_df
             self.results["Com RBU"] = ubi_df
             return


        # --- Cenário Baseline (Sem RBU) ---
        baseline_df = self._initialize_scenario()
        for year in self.years:
            prev_year = year - 1
            if prev_year not in baseline_df.index: continue # Skip if previous year data is missing

            # Copia valores do ano anterior como base para muitos indicadores baseline
            baseline_df.loc[year] = baseline_df.loc[prev_year].copy()

            # --- Cálculos do Ano Atual (Baseline) ---
            prev_pop = baseline_df.loc[prev_year, "Population (Millions)"]
            prev_mig = baseline_df.loc[prev_year, "Net Migration (Thousands)"]
            baseline_df.loc[year, "Population (Millions)"] = prev_pop * (1.0 + self.params["baseline_annual_pop_growth_rate"]) + (prev_mig / 1000.0)

            prev_prod_level = baseline_df.loc[prev_year, "Productivity Level (Index)"]
            baseline_df.loc[year, "Productivity Growth Rate"] = self.params["baseline_annual_productivity_growth_rate"]
            baseline_df.loc[year, "Productivity Level (Index)"] = prev_prod_level * (1.0 + baseline_df.loc[year, "Productivity Growth Rate"])

            current_pop = baseline_df.loc[year, "Population (Millions)"]
            current_part_rate = baseline_df.loc[year, "Labor Participation Rate"] # Mantido do ano anterior
            current_unemp_rate = baseline_df.loc[year, "Unemployment Rate"] # Mantido do ano anterior
            if pd.notna(current_pop) and pd.notna(current_part_rate) and pd.notna(current_unemp_rate) and current_pop > 0:
                potential_labor_force = current_pop * current_part_rate
                employed_labor_force = potential_labor_force * (1.0 - current_unemp_rate)
                baseline_df.loc[year, "Labor Force (Millions)"] = employed_labor_force
            else:
                 baseline_df.loc[year, "Labor Force (Millions)"] = 0.0

            gdp_prev = baseline_df.loc[prev_year, "GDP Real (Trillion EUR)"]
            prev_labor = baseline_df.loc[prev_year, "Labor Force (Millions)"]
            current_labor = baseline_df.loc[year, "Labor Force (Millions)"]
            prev_hours_b = baseline_df.loc[prev_year, "Avg Hours Worked"]
            current_hours_b = baseline_df.loc[year, "Avg Hours Worked"] # Mantido do ano anterior
            current_prod_growth_b = baseline_df.loc[year, "Productivity Growth Rate"]

            # Calcula GDP baseado em trabalho e produtividade
            gdp_current = 0.0
            if pd.notna(gdp_prev) and gdp_prev > 0:
                 labor_input_prev = prev_labor * prev_hours_b
                 labor_input_curr = current_labor * current_hours_b
                 if pd.notna(labor_input_prev) and labor_input_prev > 0 and pd.notna(labor_input_curr) and pd.notna(current_prod_growth_b):
                     labor_input_growth = (labor_input_curr / labor_input_prev) - 1.0
                     gdp_current = gdp_prev * (1.0 + labor_input_growth + current_prod_growth_b)
                 else: # Fallback se dados de trabalho ausentes
                     gdp_current = gdp_prev * (1.0 + self.params["baseline_annual_gdp_growth_rate"])

            baseline_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current if gdp_current > 0 else 0.0
            baseline_df.loc[year, "GDP Growth Rate"] = (gdp_current / gdp_prev - 1.0) if pd.notna(gdp_prev) and gdp_prev > 0 else 0.0

            # --- Baseline Social/Outros ---
            baseline_df.loc[year, "Millionaires (Thousands)"] *= 1.02
            if pd.notna(gdp_current) and gdp_current > 0 and pd.notna(gdp_prev) and gdp_prev > 0:
                 baseline_df.loc[year, "Govt Debt/GDP Ratio"] = baseline_df.loc[prev_year, "Govt Debt/GDP Ratio"] * (gdp_prev / gdp_current)
            # else: Mantem a dívida do ano anterior se GDP for zero ou NaN

            baseline_df.loc[year, "Capital Flight Index"] = 100.0
            baseline_df.loc[year, "UBI Cost (Billion EUR)"] = 0.0
            baseline_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0.0

            # --- Baseline Consumption/Investment ---
            prev_cons_b = baseline_df.loc[prev_year, "Consumption (Trillion EUR)"]
            prev_inv_b = baseline_df.loc[prev_year, "Investment (Trillion EUR)"]
            gdp_growth_b = baseline_df.loc[year, "GDP Growth Rate"]
            if pd.notna(gdp_current) and gdp_current > 0 and pd.notna(prev_cons_b) and pd.notna(prev_inv_b):
                baseline_df.loc[year, "Consumption (Trillion EUR)"] = prev_cons_b * (1.0 + gdp_growth_b)
                baseline_df.loc[year, "Investment (Trillion EUR)"] = prev_inv_b * (1.0 + gdp_growth_b)
            else:
                baseline_df.loc[year, "Consumption (Trillion EUR)"] = 0.0
                baseline_df.loc[year, "Investment (Trillion EUR)"] = 0.0


            # --- Baseline Inflation ---
            prod_growth_b = baseline_df.loc[year, "Productivity Growth Rate"]
            inflation_reduction_from_prod_b = prod_growth_b * self.params["inflation_productivity_dampening_factor"]
            baseline_inflation = self.params["baseline_annual_inflation_rate"] - inflation_reduction_from_prod_b
            baseline_df.loc[year, "Inflation Rate"] = max(0.0, baseline_inflation)

        # Remove base_year e preenche NaNs remanescentes
        self.results["Baseline (Sem RBU)"] = baseline_df.drop(index=base_year, errors='ignore').fillna(0.0)


        # --- Cenário Com RBU ---
        ubi_df = self._initialize_scenario()
        for year in self.years:
            prev_year = year - 1
            if prev_year not in ubi_df.index: continue # Skip if previous year data is missing

            ubi_active = year >= int(self.params["ubi_start_year"])
            # Copia valores do ano anterior como base
            ubi_df.loc[year] = ubi_df.loc[prev_year].copy()

            # --- Cálculos do Ano Atual (Com RBU) ---
            migration_factor = self.params["migration_net_increase_factor_restricted"] if ubi_active else 1.0
            net_migration_ubi = self.params["initial_net_migration_thousands"] * migration_factor
            ubi_df.loc[year, "Net Migration (Thousands)"] = net_migration_ubi

            prev_pop_ubi = ubi_df.loc[prev_year, "Population (Millions)"]
            fertility_boost = self.params['fertility_rate_ubi_boost_pp'] if ubi_active else 0.0
            ubi_df.loc[year, "Population (Millions)"] = prev_pop_ubi * (1.0 + self.params["baseline_annual_pop_growth_rate"] + fertility_boost) + (net_migration_ubi / 1000.0)

            ubi_cost = 0.0
            current_pop_ubi = ubi_df.loc[year, "Population (Millions)"]
            if ubi_active and pd.notna(current_pop_ubi) and current_pop_ubi > 0 and self.params["ubi_annual_amount_eur"] > 0:
                eligible_pop = current_pop_ubi * self.params["ubi_eligible_population_share"]
                ubi_cost = (eligible_pop * 1_000_000.0 * self.params["ubi_annual_amount_eur"]) / 1_000_000_000.0
            ubi_df.loc[year, "UBI Cost (Billion EUR)"] = ubi_cost

            # --- Mercado de Trabalho (Com RBU) ---
            participation_rate_base_ubi = ubi_df.loc[prev_year, "Labor Participation Rate"]
            avg_hours_base_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"]
            ubi_df.loc[year, "Labor Participation Rate"] = participation_rate_base_ubi * (1.0 - self.params["labor_participation_reduction_factor"]) if ubi_active else participation_rate_base_ubi
            ubi_df.loc[year, "Avg Hours Worked"] = avg_hours_base_ubi * (1.0 - self.params["labor_hours_reduction_factor"]) if ubi_active else avg_hours_base_ubi

            # --- Produtividade (Com RBU) ---
            productivity_boost = self.params["productivity_growth_boost_pp"] if ubi_active else 0.0
            ubi_df.loc[year, "Productivity Growth Rate"] = self.params["baseline_annual_productivity_growth_rate"] + productivity_boost
            ubi_df.loc[year, "Productivity Level (Index)"] = ubi_df.loc[prev_year, "Productivity Level (Index)"] * (1.0 + ubi_df.loc[year, "Productivity Growth Rate"])

            # --- Força de Trabalho (Com RBU) ---
            current_part_rate_ubi = ubi_df.loc[year, "Labor Participation Rate"]
            current_unemp_rate_ubi = ubi_df.loc[year, "Unemployment Rate"] # Mantido baseline
            if pd.notna(current_pop_ubi) and pd.notna(current_part_rate_ubi) and pd.notna(current_unemp_rate_ubi) and current_pop_ubi > 0:
                 potential_labor_force_ubi = current_pop_ubi * current_part_rate_ubi
                 employed_labor_force_ubi = potential_labor_force_ubi * (1.0 - current_unemp_rate_ubi)
                 ubi_df.loc[year, "Labor Force (Millions)"] = employed_labor_force_ubi
            else:
                 ubi_df.loc[year, "Labor Force (Millions)"] = 0.0

            # --- GDP (Com RBU) ---
            gdp_prev_ubi = ubi_df.loc[prev_year, "GDP Real (Trillion EUR)"]
            prev_labor_ubi = ubi_df.loc[prev_year, "Labor Force (Millions)"]
            prev_hours_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"]
            current_prod_growth_ubi = ubi_df.loc[year, "Productivity Growth Rate"]
            current_labor_ubi = ubi_df.loc[year, "Labor Force (Millions)"]
            current_hours_ubi = ubi_df.loc[year, "Avg Hours Worked"]

            # --> Variável unificada é gdp_current_ubi <--
            gdp_current_ubi = 0.0 # Inicializa
            if pd.notna(gdp_prev_ubi) and gdp_prev_ubi > 0:
                 labor_input_prev_ubi = prev_labor_ubi * prev_hours_ubi
                 labor_input_curr_ubi = current_labor_ubi * current_hours_ubi
                 if pd.notna(labor_input_prev_ubi) and labor_input_prev_ubi > 0 and pd.notna(labor_input_curr_ubi) and pd.notna(current_prod_growth_ubi):
                     labor_input_growth_ubi = (labor_input_curr_ubi / labor_input_prev_ubi) - 1.0
                     gdp_current_ubi = gdp_prev_ubi * (1.0 + labor_input_growth_ubi + current_prod_growth_ubi)
                 else: # Fallback
                     gdp_current_ubi = gdp_prev_ubi * (1.0 + self.params["baseline_annual_gdp_growth_rate"])

            ubi_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current_ubi if gdp_current_ubi > 0 else 0.0
            ubi_df.loc[year, "GDP Growth Rate"] = (gdp_current_ubi / gdp_prev_ubi - 1.0) if pd.notna(gdp_prev_ubi) and gdp_prev_ubi > 0 else 0.0

            # --- Financiamento e Dívida (Com RBU) ---
            tax_revenue = 0.0
            debt_increase_ratio = 0.0
            # --> Usa gdp_current_ubi consistentemente <--
            if ubi_active and pd.notna(gdp_current_ubi) and gdp_current_ubi > 0 and ubi_cost > 0:
                if self.params["financing_model"] in ["progressive_tax", "wealth_tax", "mixed_taxes_debt"]:
                    tax_revenue = gdp_current_ubi * 1000.0 * self.params["additional_tax_rate_for_ubi"]
                ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = tax_revenue
                unfunded_cost = ubi_cost - tax_revenue
                # Permite dívida diminuir se houver superávit (unfunded_cost < 0)
                if self.params["financing_model"] in ["debt", "mixed_taxes_debt"]:
                    debt_increase_ratio = (unfunded_cost / 1000.0) / gdp_current_ubi
            else:
                 ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0.0

            prev_debt_ratio_ubi = ubi_df.loc[prev_year, "Govt Debt/GDP Ratio"]
            current_debt_ratio = prev_debt_ratio_ubi # Default value
            if pd.notna(gdp_current_ubi) and gdp_current_ubi > 0:
                 if pd.notna(gdp_prev_ubi) and gdp_prev_ubi > 0:
                     current_debt_ratio = (prev_debt_ratio_ubi * gdp_prev_ubi / gdp_current_ubi) + debt_increase_ratio
                 else: # Caso do primeiro ano ou se GDP anterior for zero
                     current_debt_ratio = prev_debt_ratio_ubi + debt_increase_ratio
            # Se gdp_current_ubi for 0 ou NaN, mantém a razão anterior (evita divisão por zero)

            ubi_df.loc[year, "Govt Debt/GDP Ratio"] = max(0.0, current_debt_ratio)

            # --- Consumo e Investimento (Com RBU) ---
            prev_consumption_ubi = ubi_df.loc[prev_year, "Consumption (Trillion EUR)"]
            prev_investment_ubi = ubi_df.loc[prev_year, "Investment (Trillion EUR)"]
            gdp_growth_ubi = ubi_df.loc[year, "GDP Growth Rate"]
            consumption_ubi = 0.0
            investment_ubi = 0.0
            if pd.notna(prev_consumption_ubi) and pd.notna(prev_investment_ubi):
                 consumption_ubi = prev_consumption_ubi * (1.0 + gdp_growth_ubi)
                 investment_ubi = prev_investment_ubi * (1.0 + gdp_growth_ubi)

            net_consumption_boost_trillions = 0.0
            if ubi_active and ubi_cost > 0:
                consumption_boost_from_ubi = ubi_cost * self.params["consumption_propensity_ubi_recipients"]
                consumption_reduction_from_taxes = (tax_revenue * self.params["consumption_reduction_high_income_financing"]) if self.params["financing_model"] != "debt" else 0.0
                net_consumption_boost_billions = consumption_boost_from_ubi - consumption_reduction_from_taxes
                net_consumption_boost_trillions = net_consumption_boost_billions / 1000.0
                consumption_ubi += net_consumption_boost_trillions
                if self.params["financing_model"] != "debt":
                     investment_ubi *= (1.0 - self.params["additional_tax_rate_for_ubi"] * 0.1)

            ubi_df.loc[year, "Consumption (Trillion EUR)"] = max(0.0, consumption_ubi)
            ubi_df.loc[year, "Investment (Trillion EUR)"] = max(0.0, investment_ubi)

            # --- Indicadores Sociais (Com RBU) ---
            gini_base_ubi = ubi_df.loc[prev_year, "Gini Coefficient"]
            poverty_base_ubi = ubi_df.loc[prev_year, "Poverty Rate"]
            wellbeing_base_ubi = ubi_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"]
            mental_health_base_ubi = ubi_df.loc[prev_year, "Mental Health Prevalence"]
            crime_base_ubi = ubi_df.loc[prev_year, "Crime Rate Index"]
            unpaid_work_base_ubi = ubi_df.loc[prev_year, "Unpaid Work Index"]

            if ubi_active:
                ubi_amount_effect = self.params["ubi_annual_amount_eur"] / 14400.0 if self.params["ubi_annual_amount_eur"] > 0 else 0.0
                gini_reduction_max = 0.04
                gini_reduction = gini_reduction_max * ubi_amount_effect
                gini_ubi = max(0.0, gini_base_ubi - gini_reduction)
                ubi_df.loc[year, "Gini Coefficient"] = gini_ubi

                poverty_reduction_factor = 0.9
                poverty_ubi = poverty_base_ubi * (1.0 - poverty_reduction_factor * ubi_amount_effect)
                ubi_df.loc[year, "Poverty Rate"] = max(self.params.get("poverty_floor", 0.03), poverty_ubi)

                ubi_df.loc[year, "Wellbeing Satisfaction (0-10)"] = min(10.0, wellbeing_base_ubi + self.params["wellbeing_satisfaction_boost"] * ubi_amount_effect)

                mental_health_reduction = self.params["mental_health_reduction_pct"] / 100.0
                prevalence_factor = (1.0 - mental_health_reduction)
                effective_factor = 1.0 - (1.0 - prevalence_factor) * ubi_amount_effect
                ubi_df.loc[year, "Mental Health Prevalence"] = mental_health_base_ubi * effective_factor

                gini_change = max(0.0, gini_base_ubi - gini_ubi) # Garante não negativo
                crime_reduction_factor = (gini_change / 0.01) * self.params["crime_reduction_per_gini_point"] if gini_change > 0 else 0.0
                ubi_df.loc[year, "Crime Rate Index"] = max(0.0, crime_base_ubi * (1.0 - crime_reduction_factor))

                ubi_df.loc[year, "Unpaid Work Index"] = unpaid_work_base_ubi * self.params["unpaid_work_increase_factor"]

            # --- Riqueza e Fuga de Capital (Com RBU) ---
            millionaires_base_ubi = ubi_df.loc[prev_year, "Millionaires (Thousands)"]
            capital_flight_base_ubi = ubi_df.loc[prev_year, "Capital Flight Index"]
            millionaires_ubi = millionaires_base_ubi * 1.02 # Crescimento base
            capital_flight_ubi = capital_flight_base_ubi # Mantem base se não houver fuga

            if ubi_active and self.params["financing_model"] != "debt" and self.params["capital_flight_factor_high_tax"] > 0:
                flight_increase_factor = 1.0 + self.params["capital_flight_factor_high_tax"]
                capital_flight_ubi = capital_flight_base_ubi * flight_increase_factor
                millionaires_ubi *= (1.0 - self.params["capital_flight_factor_high_tax"])

            ubi_df.loc[year, "Millionaires (Thousands)"] = max(0.0, millionaires_ubi)
            ubi_df.loc[year, "Capital Flight Index"] = capital_flight_ubi if capital_flight_ubi > 0 else 0.0

            # --- Inflação (Com RBU) ---
            current_inflation = self.params["baseline_annual_inflation_rate"]
            current_prod_growth_ubi = ubi_df.loc[year, "Productivity Growth Rate"]
            if pd.notna(current_prod_growth_ubi):
                 inflation_reduction_from_prod = current_prod_growth_ubi * self.params["inflation_productivity_dampening_factor"]
                 current_inflation -= inflation_reduction_from_prod

            # --> Usa gdp_current_ubi consistentemente <--
            if ubi_active and pd.notna(gdp_current_ubi) and gdp_current_ubi > 0 and net_consumption_boost_trillions > 0:
                ubi_consumption_boost_ratio = net_consumption_boost_trillions / gdp_current_ubi
                inflation_boost_from_demand = ubi_consumption_boost_ratio * self.params["inflation_ubi_demand_sensitivity"]
                current_inflation += inflation_boost_from_demand

            ubi_df.loc[year, "Inflation Rate"] = max(0.0, current_inflation)

        # Remove base_year e preenche NaNs remanescentes
        self.results["Com RBU"] = ubi_df.drop(index=base_year, errors='ignore').fillna(0.0)


    def get_results(self):
        """Retorna os DataFrames de resultados para ambos os cenários."""
        return self.results

    def get_summary_dataframe(self, year):
        """Retorna um DataFrame NÃO FORMATADO para a tabela comparativa de um ano específico."""
        if not self.results or "Baseline (Sem RBU)" not in self.results or "Com RBU" not in self.results :
             return pd.DataFrame(columns=["Cenário Sem RBU", "Cenário Com RBU"], index=pd.Index([], name="Indicador"))

        # Garante que year é int
        target_year = int(max(int(self.params["start_year"]), min(int(year), int(self.params["end_year"]))))

        if target_year not in self.results["Baseline (Sem RBU)"].index or target_year not in self.results["Com RBU"].index:
             # st.error(f"Ano {target_year} fora do intervalo da simulação ({self.params['start_year']}-{self.params['end_year']}) ou dados ausentes.")
             # Retorna DataFrame vazio com índice para evitar erros posteriores
             return pd.DataFrame(columns=["Cenário Sem RBU", "Cenário Com RBU"], index=pd.Index([], name="Indicador"))

        baseline_data = self.results["Baseline (Sem RBU)"].loc[target_year]
        ubi_data = self.results["Com RBU"].loc[target_year]

        # Garante que os índices sejam os mesmos para evitar desalinhamento
        common_index = baseline_data.index.intersection(ubi_data.index)
        summary = pd.DataFrame({
            "Indicador": common_index, # Usa apenas indicadores comuns
            "Cenário Sem RBU": baseline_data.loc[common_index].values,
            "Cenário Com RBU": ubi_data.loc[common_index].values
        })

        return summary.set_index("Indicador") # Retorna DF com dados numéricos



# --- Interface Streamlit ---

st.set_page_config(layout="wide")
st.title("Simulador Interativo de Impactos da Renda Básica Universal (RBU)")
st.markdown("""
Use a barra lateral para ajustar **todas** as premissas da simulação, agrupadas por categoria.
Analise os impactos econômicos, sociais e fiscais de longo prazo comparando o cenário 'Com RBU' vs. 'Sem RBU' (Baseline).
**Nota:** Este é um modelo simplificado para fins ilustrativos. Os resultados dependem fortemente das premissas escolhidas.
""")

# --- Barra Lateral para Controles ---
st.sidebar.header("⚙️ Parâmetros da Simulação")

current_params = {} # Dicionário para guardar parâmetros da UI

# --- Grupo: Configuração da Simulação (AGORA COM INPUTS) ---
with st.sidebar.expander("⏱️ Configuração da Simulação", expanded=True):
    # Define valores padrão seguros
    default_start = int(DEFAULT_PARAMS["start_year"])
    default_end = int(DEFAULT_PARAMS["end_year"])

    # Inputs para ano de início e fim
    sim_start_year = st.number_input(
        "Ano Inicial da Simulação",
        min_value=2020,
        max_value=default_end - 1, # Ano inicial deve ser menor que o final
        value=default_start,
        step=1,
        key="sim_start_year_input",
        help="O primeiro ano a ser simulado."
    )
    sim_end_year = st.number_input(
        "Ano Final da Simulação",
        min_value=sim_start_year, # Ano final >= ano inicial
        max_value=2150,
        value=default_end,
        step=1,
        key="sim_end_year_input",
        help="O último ano a ser simulado."
    )

    # Validação simples
    if sim_end_year < sim_start_year:
        st.sidebar.error("O ano final deve ser maior ou igual ao ano inicial.")
        # Mantém os defaults em caso de erro para evitar quebrar o resto
        sim_start_year = default_start
        sim_end_year = default_end

    # Armazena nos parâmetros atuais
    current_params["start_year"] = int(sim_start_year)
    current_params["end_year"] = int(sim_end_year)

# --- Grupo: Configuração da RBU (AGORA DEPENDE DOS ANOS DE SIMULAÇÃO) ---
with st.sidebar.expander("🔵 Configuração da RBU", expanded=True):
    current_params["ubi_annual_amount_eur"] = st.slider(
        "Valor Anual da RBU (€/pessoa)", 0.0, 30000.0, float(DEFAULT_PARAMS["ubi_annual_amount_eur"]), 500.0, format="€%.0f", key="ubi_amount"
    )
    # Ajusta o range e o valor default do slider do ano de início da RBU
    # com base nos anos de simulação escolhidos pelo usuário
    _sim_start = current_params["start_year"]
    _sim_end = current_params["end_year"]
    _default_ubi_start = int(DEFAULT_PARAMS["ubi_start_year"])
    # Garante que o default esteja dentro do range da simulação
    ubi_start_value = max(_sim_start, min(_sim_end, _default_ubi_start))

    current_params["ubi_start_year"] = st.slider(
        "Ano de Início da RBU",
        min_value=_sim_start,  # Mínimo é o início da simulação
        max_value=_sim_end,    # Máximo é o fim da simulação
        value=ubi_start_value, # Usa valor default ajustado
        step=1,
        key="ubi_start"
    )
    current_params["ubi_eligible_population_share"] = st.slider(
        "Parcela da População Elegível (%)", 0.0, 100.0, DEFAULT_PARAMS["ubi_eligible_population_share"] * 100.0, 5.0, format="%.0f%%", key="ubi_eligibility"
    ) / 100.0

# --- Restante dos Grupos da Sidebar (com ajustes para float onde necessário) ---

# --- Grupo: Impactos Econômicos e Comportamentais ---
with st.sidebar.expander("💼 Econômicos e Comportamentais"):
    current_params["labor_participation_reduction_factor"] = st.slider(
        "Redução na Participação no Trabalho (%)", 0.0, 20.0, DEFAULT_PARAMS["labor_participation_reduction_factor"] * 100.0, 0.5, format="%.1f%%", key="labor_part_reduc"
    ) / 100.0
    current_params["labor_hours_reduction_factor"] = st.slider(
        "Redução nas Horas Médias Trabalhadas (%)", 0.0, 10.0, DEFAULT_PARAMS["labor_hours_reduction_factor"] * 100.0, 0.5, format="%.1f%%", key="labor_hours_reduc"
    ) / 100.0
    current_params["productivity_growth_boost_pp"] = st.slider(
        "Impulso Adicional na Produtividade (p.p./ano)", 0.0, 1.0, DEFAULT_PARAMS["productivity_growth_boost_pp"]*100.0, 0.05, format="%.2f p.p.", key="prod_boost"
    ) / 100.0
    current_params["consumption_propensity_ubi_recipients"] = st.slider(
        "Propensão a Consumir RBU (Beneficiários, %)", 0.0, 100.0, DEFAULT_PARAMS["consumption_propensity_ubi_recipients"] * 100.0, 5.0, format="%.0f%%", key="consump_prop_ubi"
    ) / 100.0
    current_params["consumption_reduction_high_income_financing"] = st.slider(
        "Redução Consumo Ricos (se taxados, % do imposto)", 0.0, 50.0, DEFAULT_PARAMS["consumption_reduction_high_income_financing"] * 100.0, 5.0, format="%.0f%%", key="consump_reduc_tax"
    ) / 100.0

# --- Grupo: Impactos Sociais ---
with st.sidebar.expander("🌍 Impactos Sociais"):
    current_params["wellbeing_satisfaction_boost"] = st.slider(
        "Aumento na Satisfação Média (0-10)", 0.0, 2.0, float(DEFAULT_PARAMS["wellbeing_satisfaction_boost"]), 0.1, key="wellbeing_boost"
    )
    current_params["mental_health_reduction_pct"] = st.slider(
        "Redução Prevalência Saúde Mental (%)", 0.0, 100.0, float(DEFAULT_PARAMS["mental_health_reduction_pct"]), 5.0, format="%.0f%%", key="mental_health_reduc_pct"
    )
    current_params["crime_reduction_per_gini_point"] = st.slider(
        "Redução Crime por Ponto Gini (0.01) (%)", 0.0, 10.0, DEFAULT_PARAMS["crime_reduction_per_gini_point"] * 100.0, 0.5, format="%.1f%%", key="crime_gini_reduc"
    ) / 100.0
    current_params["unpaid_work_increase_factor"] = st.slider(
        "Aumento Trabalho Não Remunerado (Índice, 1=base)", 1.0, 1.5, float(DEFAULT_PARAMS["unpaid_work_increase_factor"]), 0.01, format="%.2f", key="unpaid_work_factor"
    )

# --- Grupo: Financiamento e Reações ---
with st.sidebar.expander("💰 Financiamento e Reações"):
    financing_options = ["mixed_taxes_debt", "progressive_tax", "wealth_tax", "debt"]
    default_finance_idx = financing_options.index(DEFAULT_PARAMS["financing_model"]) if DEFAULT_PARAMS["financing_model"] in financing_options else 0
    current_params["financing_model"] = st.selectbox(
        "Modelo de Financiamento Principal", options=financing_options, index=default_finance_idx, key="financing_model"
    )
    current_params["additional_tax_rate_for_ubi"] = st.slider(
        "Alíquota Adicional Média (se financiado por imposto, % do PIB)", 0.0, 15.0, DEFAULT_PARAMS["additional_tax_rate_for_ubi"] * 100.0, 0.5, format="%.1f%%", key="tax_rate_ubi"
    ) / 100.0
    current_params["capital_flight_factor_high_tax"] = st.slider(
        "Fuga de Capital Anual (se impostos altos, % da riqueza)", 0.0, 5.0, DEFAULT_PARAMS["capital_flight_factor_high_tax"] * 100.0, 0.1, format="%.2f%%", key="capital_flight"
    ) / 100.0
    current_params["migration_net_increase_factor_restricted"] = st.slider(
        "Aumento Imigração Líquida (c/ regras, Índice 1=base)", 1.0, 2.0, float(DEFAULT_PARAMS["migration_net_increase_factor_restricted"]), 0.05, format="%.2f", key="migration_factor"
    )

# --- Grupo: Inflação ---
with st.sidebar.expander("📈 Parâmetros de Inflação"):
    current_params["baseline_annual_inflation_rate"] = st.slider(
        "Inflação Anual Base (%)", 0.0, 50.0, DEFAULT_PARAMS["baseline_annual_inflation_rate"] * 100.0, 0.5, format="%.1f%%", key="base_inflation"
    ) / 100.0
    current_params["inflation_ubi_demand_sensitivity"] = st.slider(
        "Sensibilidade Inflação à Demanda RBU", 0.0, 0.5, float(DEFAULT_PARAMS["inflation_ubi_demand_sensitivity"]), 0.01, format="%.2f", key="inflation_demand_sens"
    )
    current_params["inflation_productivity_dampening_factor"] = st.slider(
        "Redução Inflação por Produtividade", 0.0, 1.0, float(DEFAULT_PARAMS["inflation_productivity_dampening_factor"]), 0.05, format="%.2f", key="inflation_prod_damp"
    )

# --- Grupo: Impacto Populacional Adicional ---
with st.sidebar.expander("👶 Impacto Populacional"):
    current_params["fertility_rate_ubi_boost_pp"] = st.slider(
        "Aumento Taxa Cresc. Pop. por Fertilidade (p.p./ano)", 0.0, 100.0, DEFAULT_PARAMS["fertility_rate_ubi_boost_pp"]*100.0, 0.01, format="%.2f p.p.", key="fertility_boost"
    ) / 100.0

# --- Grupo: Dados Iniciais (Avançado) ---
with st.sidebar.expander("📊 Dados Iniciais (Avançado)"):
    st.caption("Ajustar com cautela. Impactam todo o cenário base.")
    current_params["initial_population_millions"] = st.number_input("População Inicial (Milhões)", min_value=1.0, value=float(DEFAULT_PARAMS["initial_population_millions"]), step=0.1, key="init_pop")
    current_params["initial_gdp_trillion_eur"] = st.number_input("PIB Inicial (Trilhões €)", min_value=0.1, value=float(DEFAULT_PARAMS["initial_gdp_trillion_eur"]), step=0.1, key="init_gdp")
    current_params["initial_participation_rate"] = st.slider("Taxa de Participação Inicial (%)", 40.0, 90.0, DEFAULT_PARAMS["initial_participation_rate"]*100.0, 1.0, format="%.0f%%", key="init_part_rate") / 100.0
    current_params["initial_avg_hours_worked_per_worker"] = st.number_input("Horas Médias Anuais Iniciais / Trabalhador", min_value=1000, max_value=2500, value=int(DEFAULT_PARAMS["initial_avg_hours_worked_per_worker"]), step=10, key="init_avg_hours")
    current_params["initial_unemployment_rate"] = st.slider("Taxa de Desemprego Inicial (%)", 1.0, 20.0, DEFAULT_PARAMS["initial_unemployment_rate"]*100.0, 0.5, format="%.1f%%", key="init_unemp_rate") / 100.0
    current_params["initial_gini_coefficient"] = st.slider("Coeficiente Gini Inicial", 0.2, 0.6, float(DEFAULT_PARAMS["initial_gini_coefficient"]), 0.01, format="%.2f", key="init_gini")
    current_params["initial_poverty_rate"] = st.slider("Taxa de Pobreza Inicial (%)", 1.0, 50.0, DEFAULT_PARAMS["initial_poverty_rate"]*100.0, 1.0, format="%.0f%%", key="init_poverty") / 100.0
    current_params["initial_wellbeing_satisfaction"] = st.slider("Satisfação Inicial (0-10)", 0.0, 10.0, float(DEFAULT_PARAMS["initial_wellbeing_satisfaction"]), 0.1, key="init_wellbeing")
    current_params["initial_mental_health_prevalence"] = st.slider("Prevalência Saúde Mental Inicial (%)", 1.0, 30.0, DEFAULT_PARAMS["initial_mental_health_prevalence"]*100.0, 1.0, format="%.0f%%", key="init_mental") / 100.0
    current_params["initial_crime_rate_index"] = st.number_input("Índice Criminalidade Inicial", min_value=10.0, value=float(DEFAULT_PARAMS["initial_crime_rate_index"]), step=5.0, key="init_crime")
    current_params["initial_net_migration_thousands"] = st.number_input("Migração Líquida Inicial (Milhares)", min_value=-500.0, max_value=1000.0, value=float(DEFAULT_PARAMS["initial_net_migration_thousands"]), step=10.0, key="init_migration")
    current_params["initial_millionaires_thousands"] = st.number_input("Milionários Iniciais (Milhares)", min_value=0.0, value=float(DEFAULT_PARAMS["initial_millionaires_thousands"]), step=10.0, key="init_millionaires")
    current_params["initial_govt_debt_gdp_ratio"] = st.slider("Dívida/PIB Inicial (%)", 0.0, 200.0, DEFAULT_PARAMS["initial_govt_debt_gdp_ratio"]*100.0, 5.0, format="%.0f%%", key="init_debt_ratio") / 100.0
    current_params["baseline_annual_pop_growth_rate"] = st.slider("Cresc. Pop. Anual Base (%)", -1.0, 2.0, DEFAULT_PARAMS["baseline_annual_pop_growth_rate"]*100.0, 0.1, format="%.2f%%", key="base_pop_growth") / 100.0
    current_params["baseline_annual_gdp_growth_rate"] = st.slider("Cresc. PIB Real Anual Base (%)", -1.0, 5.0, DEFAULT_PARAMS["baseline_annual_gdp_growth_rate"]*100.0, 0.1, format="%.2f%%", key="base_gdp_growth") / 100.0
    current_params["baseline_annual_productivity_growth_rate"] = st.slider("Cresc. Produtividade Anual Base (%)", 0.0, 3.0, DEFAULT_PARAMS["baseline_annual_productivity_growth_rate"]*100.0, 0.1, format="%.2f%%", key="base_prod_growth") / 100.0
    current_params["poverty_floor"] = st.slider("Piso Taxa de Pobreza (%)", 0.0, 10.0, DEFAULT_PARAMS["poverty_floor"]*100.0, 0.5, format="%.1f%%", key="poverty_floor") / 100.0

# Adiciona params default que não estão na sidebar (se houver)
# Garante tipos corretos para passagem ao simulador
final_params = {}
for key, value in DEFAULT_PARAMS.items():
    # Pega o valor da UI se existir, senão usa o default
    param_value = current_params.get(key, value)
    try:
        # Tenta converter para o tipo do default
        if isinstance(value, int) and key not in ['ubi_annual_amount_eur']: # Mantem amount como float
             final_params[key] = int(param_value)
        elif isinstance(value, float):
             final_params[key] = float(param_value)
        elif isinstance(value, str):
             final_params[key] = str(param_value)
        else: # Mantem outros tipos
            final_params[key] = param_value
    except (ValueError, TypeError):
         st.warning(f"Erro ao converter parâmetro '{key}'. Usando valor default.")
         final_params[key] = value # Usa default em caso de erro


# --- Execução da Simulação com Parâmetros Atuais ---
@st.cache_data
def run_sim(params_tuple): # Cache requer argumentos hashable como tupla
    params_dict = dict(params_tuple)
    # Não precisa mais de conversão de tipo aqui, já foi feito acima
    simulator = UBISimulator(params=params_dict)
    try:
        simulator.run_simulation()
        return simulator
    except Exception as e:
        st.error(f"Erro durante a execução da simulação: {e}")
        # Retorna um simulador 'vazio' para evitar quebrar o resto da UI
        empty_simulator = UBISimulator(params=params_dict)
        empty_simulator.results = {
            "Baseline (Sem RBU)": pd.DataFrame(columns=formatters_dict.keys()),
            "Com RBU": pd.DataFrame(columns=formatters_dict.keys())
        }
        return empty_simulator


# Converte dict final para tupla de itens ordenados para ser hashable pelo cache
try:
    # Tenta converter valores complexos (como numpy arrays se existirem) para tipos básicos
    items_to_tuple = []
    for k, v in final_params.items():
        if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series)):
             items_to_tuple.append((k, str(v))) # Converte para string como fallback
        else:
             items_to_tuple.append((k, v))
    params_tuple = tuple(sorted(items_to_tuple))
except TypeError as e:
    st.error(f"Erro ao criar tupla de parâmetros para cache: {e}. Alguns parâmetros podem não ser 'hashable'.")
    # Tenta converter todos os valores para string como fallback extremo
    params_tuple = tuple(sorted({k: str(v) for k,v in final_params.items()}.items()))


simulator = run_sim(params_tuple)
results = simulator.get_results()

# --- Exibição dos Resultados ---

# Verifica se os resultados são DataFrames válidos
if not isinstance(results, dict) or \
   "Baseline (Sem RBU)" not in results or not isinstance(results["Baseline (Sem RBU)"], pd.DataFrame) or \
   "Com RBU" not in results or not isinstance(results["Com RBU"], pd.DataFrame):
    st.error("Resultados da simulação inválidos ou não foram gerados.")
    st.stop()


baseline_df = results["Baseline (Sem RBU)"]
ubi_df = results["Com RBU"]

# Verifica se DataFrames não estão vazios antes de prosseguir
if baseline_df.empty or ubi_df.empty:
     st.warning("A simulação retornou resultados vazios. Verifique os parâmetros, especialmente os anos de início/fim.")
     # Cria DFs vazios com colunas para evitar erros na plotagem
     baseline_df = pd.DataFrame(columns=formatters_dict.keys(), index=pd.RangeIndex(start=final_params["start_year"], stop=final_params["end_year"]+1))
     ubi_df = pd.DataFrame(columns=formatters_dict.keys(), index=pd.RangeIndex(start=final_params["start_year"], stop=final_params["end_year"]+1))
     # Preenche com NaN ou 0 para evitar erros de plotagem
     baseline_df = baseline_df.fillna(0.0)
     ubi_df = ubi_df.fillna(0.0)
     # Não para, tenta exibir o que for possível (gráficos vazios)


st.header("📊 Resultados da Simulação")

# --- Tabela Comparativa ---
st.subheader("Tabela Comparativa Anual")
_sim_start_year_final = int(final_params["start_year"])
_sim_end_year_final = int(final_params["end_year"])
_ubi_start_year_final = int(final_params["ubi_start_year"])

# Calcula default_year com os parâmetros finais usados na simulação
try:
    default_year_table = min(_sim_end_year_final, _ubi_start_year_final + 10)
    # Garante que default_year esteja dentro do range da simulação
    default_year_table = max(_sim_start_year_final, default_year_table)
except Exception: # Fallback genérico
    default_year_table = _sim_end_year_final

# Slider para tabela
year_to_display = st.slider(
    "Selecione o ano para a tabela comparativa:",
    min_value=_sim_start_year_final,
    max_value=_sim_end_year_final,
    value=int(default_year_table),
    step=1,
    key="compare_year_slider"
)

summary_df = simulator.get_summary_dataframe(year_to_display)
if not summary_df.empty:
    st.dataframe(summary_df.style.format(formatter=formatters_dict, na_rep='-'), use_container_width=True)
else:
    st.warning(f"Não há dados disponíveis para a tabela no ano {year_to_display}.")


# --- Gráficos Comparativos ---
st.subheader("📈 Gráficos Comparativos ao Longo do Tempo")

indicators_to_plot = list(formatters_dict.keys())

sim_ubi_start_final = int(final_params["ubi_start_year"])
cols = st.columns(2)
col_idx = 0

# Garante que os índices dos dataframes existem e são compatíveis antes de plotar
if not baseline_df.index.equals(ubi_df.index):
    st.warning("Índices dos cenários Baseline e Com RBU não são idênticos. Gráficos podem estar desalinhados.")
    # Tenta reindexar para alinhar, preenchendo com NaN onde não há dados
    common_index = baseline_df.index.union(ubi_df.index)
    baseline_df = baseline_df.reindex(common_index).fillna(0.0) # Ou ffill/bfill?
    ubi_df = ubi_df.reindex(common_index).fillna(0.0)


for indicator in indicators_to_plot:
    if indicator not in baseline_df.columns or indicator not in ubi_df.columns:
        # st.warning(f"Indicador '{indicator}' não encontrado para plotagem.")
        continue # Pula indicador se não estiver presente em ambos

    chart_data_baseline = baseline_df[indicator]
    chart_data_ubi = ubi_df[indicator]

    chart_df_data = {
        'Sem RBU (Baseline)': chart_data_baseline,
        'Com RBU': chart_data_ubi
        }
    chart_data = pd.DataFrame(chart_df_data)

    # Verifica se há dados não-NaN para plotar
    if not chart_data.isnull().all().all() and not chart_data.empty:
        current_col = cols[col_idx % 2]
        with current_col:
            st.markdown(f"**{indicator}**")
            st.line_chart(chart_data)
            if sim_ubi_start_final <= _sim_end_year_final:
                 st.caption(f"Início da RBU (ano {sim_ubi_start_final}) não visualmente marcado.")
        col_idx += 1
    # else:
        # st.warning(f"Sem dados válidos para plotar o indicador '{indicator}'.")


# --- Dados Completos (para Download) ---
st.subheader("💾 Dados Completos da Simulação")
with st.expander("Ver/Baixar Tabelas de Dados"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Baseline (Sem RBU)**")
        st.dataframe(baseline_df.style.format(formatter=formatters_dict, na_rep='-'))
        st.download_button(
           "Download Baseline CSV", baseline_df.to_csv(index=True).encode('utf-8'),
           "baseline_results.csv", "text/csv", key='download-baseline'
         )
    with col2:
        st.markdown("**Cenário Com RBU**")
        st.dataframe(ubi_df.style.format(formatter=formatters_dict, na_rep='-'))
        st.download_button(
           "Download RBU CSV", ubi_df.to_csv(index=True).encode('utf-8'),
           "ubi_results.csv", "text/csv", key='download-ubi'
         )