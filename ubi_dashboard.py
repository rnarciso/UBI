import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Colar a Classe UBISimulator e DEFAULT_PARAMS do código anterior aqui ---
# (O código da classe e dos parâmetros é extenso, cole-o aqui para manter organizado)

# --- Configuração de Premissas e Parâmetros (Baseado no Texto) ---
DEFAULT_PARAMS = {
    # Parâmetros da RBU
    "ubi_annual_amount_eur": 14400,  # Ex: 1200 €/mês * 12
    "ubi_start_year": 2025,
    "ubi_eligible_population_share": 1.0, # 1.0 = Universal para todos os cidadãos/residentes elegíveis

    # Parâmetros Econômicos e Comportamentais
    "labor_participation_reduction_factor": 0.04, # Redução líquida de 4% na taxa de participação (média, texto sugere ~5%)
    "labor_hours_reduction_factor": 0.01, # Redução adicional de 1% nas horas médias por trabalhador (total 5% efeito combinado)
    "productivity_growth_boost_pp": 0.001, # Aumento de 0.1 ponto percentual no crescimento anual da produtividade
    "consumption_propensity_ubi_recipients": 0.8, # 80% da RBU gasta (classes mais baixas)
    "consumption_reduction_high_income_financing": 0.2, # Redução no consumo dos mais ricos se financiamento for por impostos sobre eles
    "innovation_startup_boost_factor": 1.05, # Aumento de 5% na taxa de criação de startups (proxy para inovação)

    # Parâmetros Sociais
    "wellbeing_satisfaction_boost": 0.5, # Aumento na satisfação média de vida (escala 0-10)
    "mental_health_improvement_factor": 0.6, # Redução de 40% na prevalência de problemas (fator 0.6)
    "crime_reduction_per_gini_point": 0.05, # Redução de 5% no crime para cada 0.01 ponto de queda no Gini
    "unpaid_work_increase_factor": 1.1, # Aumento de 10% nas horas de trabalho não remunerado (voluntariado, cuidado)

    # Parâmetros de Financiamento e Reação
    "financing_model": "mixed_taxes_debt", # Opções: "progressive_tax", "wealth_tax", "debt", "mixed_taxes_debt"
    "additional_tax_rate_for_ubi": 0.05, # Alíquota adicional média necessária (simplificado)
    "capital_flight_factor_high_tax": 0.005, # 0.5% da riqueza dos muito ricos pode sair anualmente se impostos forem altos
    "migration_net_increase_factor_restricted": 1.1, # Aumento de 10% na imigração líquida (com regras de residência)
    "migration_net_increase_factor_open": 1.5, # Aumento de 50% (hipotético, sem regras)

    # Parâmetros de Simulação
    "start_year": 2024,
    "end_year": 2075,

    # Dados Iniciais (Alemanha - Exemplo Simplificado - Usar dados reais!)
    "initial_population_millions": 84.0,
    "initial_gdp_trillion_eur": 4.0,
    "initial_participation_rate": 0.75, # Taxa de participação na força de trabalho
    "initial_avg_hours_worked_per_worker": 1600,
    "initial_unemployment_rate": 0.055,
    "initial_gini_coefficient": 0.29,
    "initial_poverty_rate": 0.15,
    "initial_wellbeing_satisfaction": 7.0,
    "initial_mental_health_prevalence": 0.10, # Ex: 10% com depressão/ansiedade
    "initial_crime_rate_index": 100, # Índice base
    "initial_net_migration_thousands": 100,
    "initial_millionaires_thousands": 150,
    "initial_govt_debt_gdp_ratio": 0.60,
    "baseline_annual_pop_growth_rate": 0.001, # Crescimento populacional anual base (pode ser negativo sem migração)
    "baseline_annual_gdp_growth_rate": 0.012, # Crescimento do PIB real anual base
    "baseline_annual_productivity_growth_rate": 0.01, # Crescimento da produtividade anual base
    "poverty_floor": 0.03 # Piso mínimo para taxa de pobreza mesmo com RBU
}


class UBISimulator:
    """
    Simulador de Impactos de Longo Prazo da Renda Básica Universal (RBU).
    Modela cenários 'Com RBU' vs. 'Sem RBU' (Baseline) ao longo de 50 anos.
    """
    def __init__(self, params=DEFAULT_PARAMS):
        self.params = params
        # Garante que end_year seja pelo menos start_year
        self.params["end_year"] = max(params["start_year"], params["end_year"])
        self.years = np.arange(params["start_year"], params["end_year"] + 1)
        self.results = {} # Armazenará os DataFrames de resultados

    def _initialize_scenario(self):
        """Cria um DataFrame para armazenar os resultados anuais de um cenário."""
        indicators = [
            # Demografia e RBU
            "Population (Millions)", "UBI Cost (Billion EUR)",
            # Mercado de Trabalho
            "Labor Participation Rate", "Avg Hours Worked", "Unemployment Rate", "Labor Force (Millions)",
            # Economia e Finanças
            "GDP Real (Trillion EUR)", "GDP Growth Rate", "Productivity Growth Rate", "Productivity Level (Index)",
            "Consumption (Trillion EUR)", "Investment (Trillion EUR)", "Govt Debt/GDP Ratio",
            "Additional Tax Revenue (Billion EUR)", "Capital Flight Index",
            # Social
            "Gini Coefficient", "Poverty Rate", "Wellbeing Satisfaction (0-10)",
            "Mental Health Prevalence", "Crime Rate Index", "Unpaid Work Index",
            # Migração e Riqueza
            "Net Migration (Thousands)", "Millionaires (Thousands)"
        ]
        # Garante que temos pelo menos o ano inicial no índice se start == end
        index_years = self.years if len(self.years) > 0 else [self.params["start_year"]]
        df = pd.DataFrame(index=index_years, columns=indicators, dtype=float)

        # Preenche valores iniciais para o ano base (start_year - 1) para cálculos
        base_year = self.params["start_year"] - 1
        df.loc[base_year, "Population (Millions)"] = self.params["initial_population_millions"]
        df.loc[base_year, "GDP Real (Trillion EUR)"] = self.params["initial_gdp_trillion_eur"]
        df.loc[base_year, "Labor Participation Rate"] = self.params["initial_participation_rate"]
        df.loc[base_year, "Avg Hours Worked"] = self.params["initial_avg_hours_worked_per_worker"]
        df.loc[base_year, "Unemployment Rate"] = self.params["initial_unemployment_rate"]
        df.loc[base_year, "Productivity Level (Index)"] = 100.0 # Base 100
        df.loc[base_year, "Gini Coefficient"] = self.params["initial_gini_coefficient"]
        df.loc[base_year, "Poverty Rate"] = self.params["initial_poverty_rate"]
        df.loc[base_year, "Wellbeing Satisfaction (0-10)"] = self.params["initial_wellbeing_satisfaction"]
        df.loc[base_year, "Mental Health Prevalence"] = self.params["initial_mental_health_prevalence"]
        df.loc[base_year, "Crime Rate Index"] = self.params["initial_crime_rate_index"]
        df.loc[base_year, "Net Migration (Thousands)"] = self.params["initial_net_migration_thousands"]
        df.loc[base_year, "Millionaires (Thousands)"] = self.params["initial_millionaires_thousands"]
        df.loc[base_year, "Govt Debt/GDP Ratio"] = self.params["initial_govt_debt_gdp_ratio"]
        df.loc[base_year, "Capital Flight Index"] = 100.0
        df.loc[base_year, "Unpaid Work Index"] = 100.0

        # Valores derivados iniciais
        # Verifica se a população é positiva antes de calcular
        pop_base = df.loc[base_year, "Population (Millions)"]
        part_rate_base = df.loc[base_year, "Labor Participation Rate"]
        unemp_rate_base = df.loc[base_year, "Unemployment Rate"]

        if pop_base > 0 and part_rate_base is not None and unemp_rate_base is not None:
             initial_labor_force = pop_base * part_rate_base
             df.loc[base_year, "Labor Force (Millions)"] = initial_labor_force * (1 - unemp_rate_base) # Empregados
        else:
             df.loc[base_year, "Labor Force (Millions)"] = 0


        # Suposições simplificadas para Consumo e Investimento iniciais como % do PIB
        gdp_base = df.loc[base_year, "GDP Real (Trillion EUR)"]
        if gdp_base > 0:
             df.loc[base_year, "Consumption (Trillion EUR)"] = gdp_base * 0.6
             df.loc[base_year, "Investment (Trillion EUR)"] = gdp_base * 0.2
        else:
             df.loc[base_year, "Consumption (Trillion EUR)"] = 0
             df.loc[base_year, "Investment (Trillion EUR)"] = 0


        return df

    def run_simulation(self):
        """Executa a simulação para os cenários Baseline e Com RBU."""
        if len(self.years) == 0: # Caso start_year == end_year
             # Apenas inicializa e retorna os dados do ano base como se fosse o único ano
             baseline_df = self._initialize_scenario().loc[[self.params["start_year"]-1]].rename(index={self.params["start_year"]-1: self.params["start_year"]})
             ubi_df = self._initialize_scenario().loc[[self.params["start_year"]-1]].rename(index={self.params["start_year"]-1: self.params["start_year"]})
             baseline_df = baseline_df.fillna(0) # Preenche NaNs com 0 para evitar erros
             ubi_df = ubi_df.fillna(0)
             self.results["Baseline (Sem RBU)"] = baseline_df
             self.results["Com RBU"] = ubi_df
             return # Não há loop para executar

        # --- Cenário Baseline (Sem RBU) ---
        baseline_df = self._initialize_scenario()
        for year in self.years:
            prev_year = year - 1

            # --- Cálculos do Ano Atual (Baseline) ---
            # Garante que valores anteriores existem e são numéricos
            prev_pop = baseline_df.loc[prev_year, "Population (Millions)"] if pd.notna(baseline_df.loc[prev_year, "Population (Millions)"]) else 0
            prev_mig = baseline_df.loc[prev_year, "Net Migration (Thousands)"] if pd.notna(baseline_df.loc[prev_year, "Net Migration (Thousands)"]) else 0
            baseline_df.loc[year, "Population (Millions)"] = prev_pop * (1 + self.params["baseline_annual_pop_growth_rate"]) + (prev_mig / 1000)
            baseline_df.loc[year, "Net Migration (Thousands)"] = prev_mig # Assume constante

            prev_part_rate = baseline_df.loc[prev_year, "Labor Participation Rate"] if pd.notna(baseline_df.loc[prev_year, "Labor Participation Rate"]) else 0
            prev_hours = baseline_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(baseline_df.loc[prev_year, "Avg Hours Worked"]) else 0
            prev_unemp = baseline_df.loc[prev_year, "Unemployment Rate"] if pd.notna(baseline_df.loc[prev_year, "Unemployment Rate"]) else 0
            baseline_df.loc[year, "Labor Participation Rate"] = prev_part_rate
            baseline_df.loc[year, "Avg Hours Worked"] = prev_hours
            baseline_df.loc[year, "Unemployment Rate"] = prev_unemp

            prev_prod_level = baseline_df.loc[prev_year, "Productivity Level (Index)"] if pd.notna(baseline_df.loc[prev_year, "Productivity Level (Index)"]) else 0
            baseline_df.loc[year, "Productivity Growth Rate"] = self.params["baseline_annual_productivity_growth_rate"]
            baseline_df.loc[year, "Productivity Level (Index)"] = prev_prod_level * (1 + baseline_df.loc[year, "Productivity Growth Rate"])

            current_pop = baseline_df.loc[year, "Population (Millions)"] if pd.notna(baseline_df.loc[year, "Population (Millions)"]) else 0
            current_part_rate = baseline_df.loc[year, "Labor Participation Rate"]
            current_unemp_rate = baseline_df.loc[year, "Unemployment Rate"]
            potential_labor_force = current_pop * current_part_rate
            employed_labor_force = potential_labor_force * (1 - current_unemp_rate)
            baseline_df.loc[year, "Labor Force (Millions)"] = employed_labor_force

            gdp_prev = baseline_df.loc[prev_year, "GDP Real (Trillion EUR)"] if pd.notna(baseline_df.loc[prev_year, "GDP Real (Trillion EUR)"]) else 0
            gdp_current = gdp_prev * (1 + self.params["baseline_annual_gdp_growth_rate"]) # Simplificado
            baseline_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current if gdp_current > 0 else 0
            baseline_df.loc[year, "GDP Growth Rate"] = (gdp_current / gdp_prev - 1) if gdp_prev else 0

            prev_gini = baseline_df.loc[prev_year, "Gini Coefficient"] if pd.notna(baseline_df.loc[prev_year, "Gini Coefficient"]) else 0
            prev_poverty = baseline_df.loc[prev_year, "Poverty Rate"] if pd.notna(baseline_df.loc[prev_year, "Poverty Rate"]) else 0
            prev_wellbeing = baseline_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"] if pd.notna(baseline_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"]) else 0
            prev_mental = baseline_df.loc[prev_year, "Mental Health Prevalence"] if pd.notna(baseline_df.loc[prev_year, "Mental Health Prevalence"]) else 0
            prev_crime = baseline_df.loc[prev_year, "Crime Rate Index"] if pd.notna(baseline_df.loc[prev_year, "Crime Rate Index"]) else 0
            prev_unpaid = baseline_df.loc[prev_year, "Unpaid Work Index"] if pd.notna(baseline_df.loc[prev_year, "Unpaid Work Index"]) else 0
            baseline_df.loc[year, "Gini Coefficient"] = prev_gini
            baseline_df.loc[year, "Poverty Rate"] = prev_poverty
            baseline_df.loc[year, "Wellbeing Satisfaction (0-10)"] = prev_wellbeing
            baseline_df.loc[year, "Mental Health Prevalence"] = prev_mental
            baseline_df.loc[year, "Crime Rate Index"] = prev_crime
            baseline_df.loc[year, "Unpaid Work Index"] = prev_unpaid

            prev_millionaires = baseline_df.loc[prev_year, "Millionaires (Thousands)"] if pd.notna(baseline_df.loc[prev_year, "Millionaires (Thousands)"]) else 0
            prev_debt_ratio = baseline_df.loc[prev_year, "Govt Debt/GDP Ratio"] if pd.notna(baseline_df.loc[prev_year, "Govt Debt/GDP Ratio"]) else 0
            baseline_df.loc[year, "Millionaires (Thousands)"] = prev_millionaires * 1.02
            baseline_df.loc[year, "Govt Debt/GDP Ratio"] = prev_debt_ratio
            baseline_df.loc[year, "Capital Flight Index"] = 100.0
            baseline_df.loc[year, "UBI Cost (Billion EUR)"] = 0
            baseline_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = 0

            current_gdp = baseline_df.loc[year, "GDP Real (Trillion EUR)"]
            baseline_df.loc[year, "Consumption (Trillion EUR)"] = current_gdp * 0.6 if current_gdp > 0 else 0
            baseline_df.loc[year, "Investment (Trillion EUR)"] = current_gdp * 0.2 if current_gdp > 0 else 0


        self.results["Baseline (Sem RBU)"] = baseline_df.drop(self.params["start_year"] - 1).fillna(0) # Remove ano base e preenche NaNs

        # --- Cenário Com RBU ---
        ubi_df = self._initialize_scenario()
        for year in self.years:
            prev_year = year - 1
            ubi_active = year >= self.params["ubi_start_year"]

            # --- Cálculos do Ano Atual (Com RBU) ---
            # Garante que valores anteriores existem e são numéricos
            migration_factor = 1.0
            if ubi_active:
                 migration_factor = self.params["migration_net_increase_factor_restricted"]
            prev_mig_ubi = ubi_df.loc[prev_year, "Net Migration (Thousands)"] if pd.notna(ubi_df.loc[prev_year, "Net Migration (Thousands)"]) else 0
            net_migration_ubi = prev_mig_ubi * migration_factor
            ubi_df.loc[year, "Net Migration (Thousands)"] = net_migration_ubi

            prev_pop_ubi = ubi_df.loc[prev_year, "Population (Millions)"] if pd.notna(ubi_df.loc[prev_year, "Population (Millions)"]) else 0
            ubi_df.loc[year, "Population (Millions)"] = prev_pop_ubi * (1 + self.params["baseline_annual_pop_growth_rate"]) + (net_migration_ubi / 1000)

            ubi_cost = 0
            current_pop_ubi = ubi_df.loc[year, "Population (Millions)"] if pd.notna(ubi_df.loc[year, "Population (Millions)"]) else 0
            if ubi_active and current_pop_ubi > 0:
                eligible_pop = current_pop_ubi * self.params["ubi_eligible_population_share"]
                ubi_cost = (eligible_pop * 1_000_000 * self.params["ubi_annual_amount_eur"]) / 1_000_000_000
            ubi_df.loc[year, "UBI Cost (Billion EUR)"] = ubi_cost

            participation_rate_base_ubi = ubi_df.loc[prev_year, "Labor Participation Rate"] if pd.notna(ubi_df.loc[prev_year, "Labor Participation Rate"]) else 0
            avg_hours_base_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(ubi_df.loc[prev_year, "Avg Hours Worked"]) else 0
            participation_rate_ubi = participation_rate_base_ubi * (1 - self.params["labor_participation_reduction_factor"]) if ubi_active else participation_rate_base_ubi
            avg_hours_ubi = avg_hours_base_ubi * (1 - self.params["labor_hours_reduction_factor"]) if ubi_active else avg_hours_base_ubi
            ubi_df.loc[year, "Labor Participation Rate"] = participation_rate_ubi
            ubi_df.loc[year, "Avg Hours Worked"] = avg_hours_ubi
            prev_unemp_ubi = ubi_df.loc[prev_year, "Unemployment Rate"] if pd.notna(ubi_df.loc[prev_year, "Unemployment Rate"]) else 0
            ubi_df.loc[year, "Unemployment Rate"] = prev_unemp_ubi

            productivity_growth_base_ubi = self.params["baseline_annual_productivity_growth_rate"]
            productivity_boost = self.params["productivity_growth_boost_pp"] if ubi_active else 0
            ubi_df.loc[year, "Productivity Growth Rate"] = productivity_growth_base_ubi + productivity_boost
            prev_prod_level_ubi = ubi_df.loc[prev_year, "Productivity Level (Index)"] if pd.notna(ubi_df.loc[prev_year, "Productivity Level (Index)"]) else 0
            ubi_df.loc[year, "Productivity Level (Index)"] = prev_prod_level_ubi * (1 + ubi_df.loc[year, "Productivity Growth Rate"])

            potential_labor_force_ubi = current_pop_ubi * participation_rate_ubi
            current_unemp_rate_ubi = ubi_df.loc[year, "Unemployment Rate"]
            employed_labor_force_ubi = potential_labor_force_ubi * (1 - current_unemp_rate_ubi)
            ubi_df.loc[year, "Labor Force (Millions)"] = employed_labor_force_ubi

            gdp_prev_ubi = ubi_df.loc[prev_year, "GDP Real (Trillion EUR)"] if pd.notna(ubi_df.loc[prev_year, "GDP Real (Trillion EUR)"]) else 0
            prev_labor_force_ubi = ubi_df.loc[prev_year, "Labor Force (Millions)"] if pd.notna(ubi_df.loc[prev_year, "Labor Force (Millions)"]) else 0
            prev_avg_hours_ubi = ubi_df.loc[prev_year, "Avg Hours Worked"] if pd.notna(ubi_df.loc[prev_year, "Avg Hours Worked"]) else 0

            if prev_labor_force_ubi > 0 and prev_avg_hours_ubi > 0 and prev_prod_level_ubi > 0:
                labor_factor_change_ubi = (employed_labor_force_ubi * avg_hours_ubi) / (prev_labor_force_ubi * prev_avg_hours_ubi)
                prod_factor_change_ubi = ubi_df.loc[year, "Productivity Level (Index)"] / prev_prod_level_ubi
            else:
                labor_factor_change_ubi = 1
                prod_factor_change_ubi = 1


            delta_labor_impact = (labor_factor_change_ubi - 1)
            delta_prod_impact = (prod_factor_change_ubi - 1)
            gdp_current_ubi = gdp_prev_ubi * (1 + self.params["baseline_annual_gdp_growth_rate"] + delta_labor_impact + delta_prod_impact + productivity_boost)
            ubi_df.loc[year, "GDP Real (Trillion EUR)"] = gdp_current_ubi if gdp_current_ubi > 0 else 0
            ubi_df.loc[year, "GDP Growth Rate"] = (gdp_current_ubi / gdp_prev_ubi - 1) if gdp_prev_ubi else 0

            tax_revenue = 0
            debt_increase_ratio = 0
            current_gdp_ubi = ubi_df.loc[year, "GDP Real (Trillion EUR)"]
            if ubi_active and current_gdp_ubi > 0:
                if self.params["financing_model"] in ["progressive_tax", "mixed_taxes_debt"]:
                    tax_revenue = current_gdp_ubi * 1000 * self.params["additional_tax_rate_for_ubi"]
                ubi_df.loc[year, "Additional Tax Revenue (Billion EUR)"] = tax_revenue

                unfunded_cost = ubi_cost - tax_revenue
                if unfunded_cost > 0 and self.params["financing_model"] in ["debt", "mixed_taxes_debt"]:
                     debt_increase_ratio = (unfunded_cost / 1000) / current_gdp_ubi

            prev_debt_ratio_ubi = ubi_df.loc[prev_year, "Govt Debt/GDP Ratio"] if pd.notna(ubi_df.loc[prev_year, "Govt Debt/GDP Ratio"]) else 0
            current_debt_ratio = (prev_debt_ratio_ubi * gdp_prev_ubi / current_gdp_ubi) + debt_increase_ratio if current_gdp_ubi > 0 else prev_debt_ratio_ubi
            ubi_df.loc[year, "Govt Debt/GDP Ratio"] = current_debt_ratio

            prev_consumption_ubi = ubi_df.loc[prev_year, "Consumption (Trillion EUR)"] if pd.notna(ubi_df.loc[prev_year, "Consumption (Trillion EUR)"]) else 0
            prev_investment_ubi = ubi_df.loc[prev_year, "Investment (Trillion EUR)"] if pd.notna(ubi_df.loc[prev_year, "Investment (Trillion EUR)"]) else 0
            consumption_ubi = prev_consumption_ubi * (1 + ubi_df.loc[year, "GDP Growth Rate"])
            investment_ubi = prev_investment_ubi * (1 + ubi_df.loc[year, "GDP Growth Rate"])

            if ubi_active:
                 net_consumption_boost = (ubi_cost * self.params["consumption_propensity_ubi_recipients"] - \
                                         tax_revenue * self.params["consumption_reduction_high_income_financing"]) / 1000
                 consumption_ubi += net_consumption_boost
                 investment_ubi *= (1 - self.params["additional_tax_rate_for_ubi"] * 0.1)

            ubi_df.loc[year, "Consumption (Trillion EUR)"] = consumption_ubi if consumption_ubi > 0 else 0
            ubi_df.loc[year, "Investment (Trillion EUR)"] = investment_ubi if investment_ubi > 0 else 0

            gini_base_ubi = ubi_df.loc[prev_year, "Gini Coefficient"] if pd.notna(ubi_df.loc[prev_year, "Gini Coefficient"]) else 0
            poverty_base_ubi = ubi_df.loc[prev_year, "Poverty Rate"] if pd.notna(ubi_df.loc[prev_year, "Poverty Rate"]) else 0
            wellbeing_base_ubi = ubi_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"] if pd.notna(ubi_df.loc[prev_year, "Wellbeing Satisfaction (0-10)"]) else 0
            mental_health_base_ubi = ubi_df.loc[prev_year, "Mental Health Prevalence"] if pd.notna(ubi_df.loc[prev_year, "Mental Health Prevalence"]) else 0
            crime_base_ubi = ubi_df.loc[prev_year, "Crime Rate Index"] if pd.notna(ubi_df.loc[prev_year, "Crime Rate Index"]) else 0
            unpaid_work_base_ubi = ubi_df.loc[prev_year, "Unpaid Work Index"] if pd.notna(ubi_df.loc[prev_year, "Unpaid Work Index"]) else 0

            if ubi_active:
                gini_reduction = 0.04 * (self.params["ubi_annual_amount_eur"] / 14400) if self.params["ubi_annual_amount_eur"] > 0 else 0
                gini_ubi = max(0, gini_base_ubi - gini_reduction)
                ubi_df.loc[year, "Gini Coefficient"] = gini_ubi

                poverty_reduction_factor = 0.9
                poverty_ubi = poverty_base_ubi * (1 - poverty_reduction_factor)
                ubi_df.loc[year, "Poverty Rate"] = max(self.params.get("poverty_floor", 0.03), poverty_ubi)

                ubi_df.loc[year, "Wellbeing Satisfaction (0-10)"] = min(10, wellbeing_base_ubi + self.params["wellbeing_satisfaction_boost"])
                ubi_df.loc[year, "Mental Health Prevalence"] = mental_health_base_ubi * self.params["mental_health_improvement_factor"]

                gini_change = gini_base_ubi - gini_ubi
                crime_reduction = (gini_change / 0.01) * self.params["crime_reduction_per_gini_point"] if gini_change > 0 else 0
                ubi_df.loc[year, "Crime Rate Index"] = crime_base_ubi * (1 - crime_reduction) if crime_base_ubi > 0 else 0

                ubi_df.loc[year, "Unpaid Work Index"] = unpaid_work_base_ubi * self.params["unpaid_work_increase_factor"]
            else:
                ubi_df.loc[year, "Gini Coefficient"] = gini_base_ubi
                ubi_df.loc[year, "Poverty Rate"] = poverty_base_ubi
                ubi_df.loc[year, "Wellbeing Satisfaction (0-10)"] = wellbeing_base_ubi
                ubi_df.loc[year, "Mental Health Prevalence"] = mental_health_base_ubi
                ubi_df.loc[year, "Crime Rate Index"] = crime_base_ubi
                ubi_df.loc[year, "Unpaid Work Index"] = unpaid_work_base_ubi

            millionaires_base_ubi = ubi_df.loc[prev_year, "Millionaires (Thousands)"] if pd.notna(ubi_df.loc[prev_year, "Millionaires (Thousands)"]) else 0
            capital_flight_base_ubi = ubi_df.loc[prev_year, "Capital Flight Index"] if pd.notna(ubi_df.loc[prev_year, "Capital Flight Index"]) else 0
            millionaires_ubi = millionaires_base_ubi * 1.02
            capital_flight_ubi = capital_flight_base_ubi

            if ubi_active and self.params["financing_model"] != "debt":
                 flight_factor = 1 + self.params["capital_flight_factor_high_tax"]
                 capital_flight_ubi = capital_flight_base_ubi * flight_factor
                 millionaires_ubi *= (1 - self.params["capital_flight_factor_high_tax"])

            ubi_df.loc[year, "Millionaires (Thousands)"] = millionaires_ubi if millionaires_ubi > 0 else 0
            ubi_df.loc[year, "Capital Flight Index"] = capital_flight_ubi if capital_flight_ubi > 0 else 0


        self.results["Com RBU"] = ubi_df.drop(self.params["start_year"] - 1).fillna(0) # Remove ano base e preenche NaNs

    def get_results(self):
        """Retorna os DataFrames de resultados para ambos os cenários."""
        return self.results

    def get_summary_dataframe(self, year):
        """Retorna um DataFrame formatado para a tabela comparativa de um ano específico."""
        if not self.results:
             # Retorna DataFrame vazio se não houver resultados
             return pd.DataFrame(columns=["Indicador", "Cenário Sem RBU", "Cenário Com RBU"])

        target_year = max(self.params["start_year"], min(year, self.params["end_year"])) # Garante que o ano está no intervalo

        if target_year not in self.results["Baseline (Sem RBU)"].index:
             st.error(f"Ano {target_year} fora do intervalo da simulação ({self.params['start_year']}-{self.params['end_year']}).")
              # Retorna DataFrame vazio
             return pd.DataFrame(columns=["Indicador", "Cenário Sem RBU", "Cenário Com RBU"])


        baseline_data = self.results["Baseline (Sem RBU)"].loc[target_year]
        ubi_data = self.results["Com RBU"].loc[target_year]

        summary = pd.DataFrame({
            "Indicador": baseline_data.index,
            "Cenário Sem RBU": baseline_data.values,
            "Cenário Com RBU": ubi_data.values
        })

        # Formatação para melhor leitura
        formatters = {
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
            "Millionaires (Thousands)": "{:.0f}"
        }

        formatted_summary = summary.copy()
        # Aplica formatação apenas se a coluna existir e não for NaN
        for indicator_name, fmt in formatters.items():
             mask = formatted_summary["Indicador"] == indicator_name
             if mask.any():
                  idx = formatted_summary[mask].index
                  for col in ["Cenário Sem RBU", "Cenário Com RBU"]:
                      # Tenta formatar, trata erros (ex: valor não numérico)
                      try:
                          formatted_summary.loc[idx, col] = formatted_summary.loc[idx, col].apply(
                              lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else ('0' if pd.notna(x) else 'N/A')
                          )
                      except (ValueError, TypeError):
                           formatted_summary.loc[idx, col] = 'Error'


        return formatted_summary.set_index("Indicador")

# --- Interface Streamlit ---

st.set_page_config(layout="wide")
st.title("Simulador Interativo de Impactos da Renda Básica Universal (RBU)")
st.markdown("Baseado nas premissas do texto fornecido. Ajuste os parâmetros na barra lateral para ver os efeitos.")

# --- Barra Lateral para Controles ---
st.sidebar.header("Parâmetros da Simulação")

# Parâmetros da RBU
st.sidebar.subheader("Configuração da RBU")
ubi_amount = st.sidebar.slider(
    "Valor Anual da RBU por pessoa (€)",
    min_value=0, max_value=25000, value=DEFAULT_PARAMS["ubi_annual_amount_eur"], step=500,
    format="€%d"
)
ubi_start = st.sidebar.slider(
    "Ano de Início da RBU",
    min_value=DEFAULT_PARAMS["start_year"], max_value=DEFAULT_PARAMS["end_year"] - 5, value=DEFAULT_PARAMS["ubi_start_year"], step=1
)

# Parâmetros Comportamentais e Econômicos
st.sidebar.subheader("Impactos no Trabalho e Produtividade")
labor_reduction = st.sidebar.slider(
    "Redução na Participação no Trabalho (%)",
    min_value=0.0, max_value=0.15, value=DEFAULT_PARAMS["labor_participation_reduction_factor"], step=0.005,
    format="%.1f%%"
)
prod_boost = st.sidebar.slider(
    "Impulso Adicional na Produtividade Anual (p.p.)",
    min_value=0.0, max_value=0.005, value=DEFAULT_PARAMS["productivity_growth_boost_pp"], step=0.0005,
    format="%.3f p.p."
)

# Parâmetros Sociais (Exemplo: Bem-estar)
st.sidebar.subheader("Impactos Sociais")
wellbeing_boost = st.sidebar.slider(
    "Aumento na Satisfação Média (0-10)",
    min_value=0.0, max_value=1.5, value=DEFAULT_PARAMS["wellbeing_satisfaction_boost"], step=0.1
)

# Parâmetros de Financiamento e Reação
st.sidebar.subheader("Financiamento e Reações")
financing = st.sidebar.selectbox(
    "Modelo de Financiamento Principal",
    options=["mixed_taxes_debt", "progressive_tax", "wealth_tax", "debt"],
    index=["mixed_taxes_debt", "progressive_tax", "wealth_tax", "debt"].index(DEFAULT_PARAMS["financing_model"])
)
capital_flight = st.sidebar.slider(
    "Fuga de Capital Anual por Impostos (%)",
    min_value=0.0, max_value=0.03, value=DEFAULT_PARAMS["capital_flight_factor_high_tax"], step=0.001,
    format="%.2f%%"
)


# --- Execução da Simulação com Parâmetros Atuais ---

# Cria dicionário com parâmetros atualizados pela interface
current_params = DEFAULT_PARAMS.copy()
current_params["ubi_annual_amount_eur"] = ubi_amount
current_params["ubi_start_year"] = ubi_start
current_params["labor_participation_reduction_factor"] = labor_reduction
current_params["productivity_growth_boost_pp"] = prod_boost
current_params["wellbeing_satisfaction_boost"] = wellbeing_boost
current_params["financing_model"] = financing
current_params["capital_flight_factor_high_tax"] = capital_flight

# Instancia e roda o simulador
# Usar cache pode ser útil para simulações mais longas, mas aqui pode ser rápido o suficiente
# @st.cache_data # Descomente se a simulação ficar lenta
def run_sim(params):
    simulator = UBISimulator(params=params)
    simulator.run_simulation()
    return simulator

simulator = run_sim(current_params)
results = simulator.get_results()

# Verifica se os resultados foram gerados
if "Baseline (Sem RBU)" not in results or "Com RBU" not in results:
    st.error("Erro ao gerar resultados da simulação. Verifique os parâmetros.")
else:
    baseline_df = results["Baseline (Sem RBU)"]
    ubi_df = results["Com RBU"]

    # --- Exibição dos Resultados ---

    st.header("Resultados da Simulação")

    # Tabela Comparativa
    st.subheader("Tabela Comparativa Anual")
    year_to_display = st.slider(
        "Selecione o ano para a tabela comparativa:",
        min_value=current_params["start_year"],
        max_value=current_params["end_year"],
        value=current_params["end_year"], # Default para o último ano
        step=1
    )
    summary_df = simulator.get_summary_dataframe(year_to_display)
    st.dataframe(summary_df, use_container_width=True) # Exibe o DataFrame formatado

    # Gráficos Comparativos
    st.subheader("Gráficos Comparativos ao Longo do Tempo")

    indicators_to_plot = [
        "GDP Real (Trillion EUR)",
        "Labor Participation Rate",
        "Gini Coefficient",
        "Poverty Rate",
        "Wellbeing Satisfaction (0-10)",
        "Govt Debt/GDP Ratio",
        "UBI Cost (Billion EUR)", # Adicionado para visualizar custo
        "Crime Rate Index"
    ]

    # Usando st.line_chart para simplicidade
    for indicator in indicators_to_plot:
        if indicator in baseline_df.columns and indicator in ubi_df.columns:
            st.markdown(f"**{indicator}**")
            # Combina as séries para o st.line_chart
            chart_data = pd.DataFrame({
                'Sem RBU (Baseline)': baseline_df[indicator],
                'Com RBU': ubi_df[indicator]
            })
            # Adiciona linha vertical para início da RBU se aplicável
            if ubi_active_in_plot := any(ubi_df.loc[year_idx, "UBI Cost (Billion EUR)"] > 0 for year_idx in ubi_df.index if year_idx >= ubi_start):
                 st.line_chart(chart_data)
                 # Nota: Adicionar linha vertical diretamente no st.line_chart não é trivial.
                 # Para isso, seria melhor usar Matplotlib ou Altair.
                 st.caption(f"Linha tracejada vermelha indicaria início da RBU em {ubi_start} (funcionalidade visual avançada)")

            else:
                 st.line_chart(chart_data)

        else:
            st.warning(f"Indicador '{indicator}' não encontrado nos resultados para plotagem.")


    # Alternativa usando Matplotlib (mais controle, mas mais código)
    # st.subheader("Gráficos Comparativos (Matplotlib)")
    # fig, axes = plt.subplots(len(indicators_to_plot), 1, figsize=(10, 5 * len(indicators_to_plot)), sharex=True)
    # if len(indicators_to_plot) == 1: axes = [axes] # Ajuste para caso de 1 plot

    # fig.suptitle("Simulação RBU: Comparativo de Cenários", fontsize=16)

    # for i, indicator in enumerate(indicators_to_plot):
    #     if indicator in baseline_df.columns and indicator in ubi_df.columns:
    #         baseline_df[indicator].plot(ax=axes[i], label="Sem RBU (Baseline)", legend=True)
    #         ubi_df[indicator].plot(ax=axes[i], label="Com RBU", legend=True)
    #         axes[i].set_ylabel(indicator)
    #         axes[i].grid(True)
    #         # Adiciona linha vertical para início da RBU
    #         if any(ubi_df.loc[year_idx, "UBI Cost (Billion EUR)"] > 0 for year_idx in ubi_df.index if year_idx >= ubi_start):
    #             axes[i].axvline(x=ubi_start, color='r', linestyle='--', linewidth=1, label=f'Início RBU ({ubi_start})')
    #             # Evita duplicar label da linha vertical na legenda
    #             handles, labels = axes[i].get_legend_handles_labels()
    #             by_label = dict(zip(labels, handles))
    #             axes[i].legend(by_label.values(), by_label.keys())


    # if len(indicators_to_plot) > 0:
    #     axes[-1].set_xlabel("Ano")
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Ajusta para título
    #     st.pyplot(fig)
    # else:
    #     st.write("Nenhum indicador selecionado para plotar.")

    # --- Adicionar mais visualizações ou análises conforme necessário ---
    st.subheader("Dados Completos (para Download)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cenário Baseline (Sem RBU)**")
        st.dataframe(baseline_df)
        st.download_button(
           "Download Baseline CSV",
           baseline_df.to_csv().encode('utf-8'),
           "baseline_results.csv",
           "text/csv",
           key='download-baseline'
         )
    with col2:
        st.markdown("**Cenário Com RBU**")
        st.dataframe(ubi_df)
        st.download_button(
           "Download RBU CSV",
           ubi_df.to_csv().encode('utf-8'),
           "ubi_results.csv",
           "text/csv",
           key='download-ubi'
         )