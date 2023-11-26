-- Checking data has come across okay
SELECT *
FROM PortfolioProject..CovidDeaths
ORDER BY 3,4

-- Checking data more granularly 
SELECT location, date, population, total_cases, new_cases
FROM PortfolioProject..CovidDeaths
ORDER BY 1,2

-- Manipulate data to show Percentage of Death for UK
SELECT location, date, population, total_deaths, (total_deaths/population)*100 AS DeathPercentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%kingdom%'
ORDER BY 1,2

-- Manipulate data to show Percentage of population who got COVID
SELECT location, date, population, total_cases, (total_cases/population)*100 AS InfectionPercentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%kingdom%'
ORDER BY 1,2

-- Manipulate data to show highest infection rate per country compared to population, using 'ORDER' to show highest across the globe
SELECT location, population, MAX(total_cases) AS HighestInfectionNumber, MAX((total_cases/population))*100 AS InfectionPercentage
FROM PortfolioProject..CovidDeaths
--WHERE location like '%kingdom%'
GROUP BY location, population
ORDER BY InfectionPercentage DESC

--Manipulate data to show highest death count per population
SELECT location, MAX(total_deaths) AS TotalDeathCount
FROM PortfolioProject..CovidDeaths
WHERE continent IS NOT NULL -- AND location like '%kingdom%'
GROUP BY location
ORDER BY TotalDeathCount DESC

--As above but break it down by continent
SELECT location, MAX(total_deaths) AS TotalDeathCountContinent
FROM PortfolioProject..CovidDeaths
WHERE continent IS NULL
GROUP BY location
ORDER BY TotalDeathCountContinent DESC

-- Manipulating data to show global statistics - Overall death percentage on a given day
SELECT date, SUM(CAST(new_cases AS float)) AS TotalNewCases, SUM(CAST(new_deaths AS float)) AS TotalNewDeaths, SUM(CAST(new_deaths AS float))/SUM(CAST(new_cases AS float))*100 AS DeathPercentage
FROM PortfolioProject..CovidDeaths
WHERE new_cases >0 AND new_deaths >0
GROUP BY date
ORDER BY 1,2

-- Total cases by the total deaths to give the total percentage
SELECT SUM(CAST(new_cases AS float)) AS Total_Cases, SUM(CAST(new_deaths AS float)) AS Total_Deaths, SUM(CAST(new_deaths AS float))/SUM(CAST(new_cases AS float))*100 AS Total_Percentage
FROM PortfolioProject..CovidDeaths
WHERE new_cases >0 AND new_deaths >0
ORDER BY 1,2

-- Creating a CTE to join the two tables and look at total population who have been vaccinated whilst creating a rolling tally and outputting that as a percentage

WITH PopulationOverVaccinations (continent, location, date, population, new_vaccinations, Rolling_Total_Vaccinations)
AS
(
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(vac.new_vaccinations) OVER (PARTITION BY dea.location ORDER BY 
dea.location, dea.DATE) AS Rolling_Total_Vaccinations
FROM PortfolioProject..CovidDeaths dea
JOIN PortfolioProject..CovidVaccinations vac
	on CAST(dea.location AS nvarchar) = CAST(vac.location AS nvarchar)
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL
--ORDER BY 2,3
)

SELECT *, (Rolling_Total_Vaccinations/population)*100
FROM PopulationOverVaccinations

-- Create the data in a view to store data for later visuals


CREATE VIEW PERVPOPVACC AS
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations, SUM(vac.new_vaccinations) OVER (PARTITION BY dea.location ORDER BY 
dea.location, dea.DATE) AS Rolling_Total_Vaccinations
FROM PortfolioProject..CovidDeaths dea
JOIN PortfolioProject..CovidVaccinations vac
	on CAST(dea.location AS nvarchar) = CAST(vac.location AS nvarchar)
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL
--ORDER BY 2,3

-- View can now be used as a table
SELECT *
FROM PERVPOPVACC