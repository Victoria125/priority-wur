# Research update demo commands for the priority-wur project.
# Read-only: displays local project results, except demo 5b (opens a web page).
# PowerShell: from dreams_molecular_families, run: . .\demo.ps1
# Then use: demo 1, demo 1b, demo 2, demo 2b, demo 3, demo 3b, demo 4, demo 5, demo 5b.

param([string]$Part)

$script:DemoRoot = $PSScriptRoot

function demo {
    param([Parameter(Mandatory = $true)][string]$n)

    $results = Join-Path $script:DemoRoot 'RESULTS.md'
    $inventory = Join-Path $script:DemoRoot '..\data\derived\podp_projects.csv'
    $links = Join-Path $script:DemoRoot '..\data\derived\podp_bgc_ms2_links.csv'
    $tutorial = Join-Path $script:DemoRoot 'reproduction\dreams_networking_tutorial.json'
    $federated = Join-Path $script:DemoRoot 'federated_prototype\RESULTS_federated_prototype.md'
    $feasibility = Join-Path $script:DemoRoot 'linking_feasibility\RESULTS_linking_feasibility.md'

    try {
        switch ($n) {
            '1' {
                $projects = @(Import-Csv $inventory | Where-Object { [double]$_.actino_fraction -ge 0.5 })
                'Project entries: ' + $projects.Count
                'Actinobacterial genomes: ' + ($projects | Measure-Object -Property n_actino_genomes -Sum).Sum
                'Genome-metabolomics file links: ' + ($projects | Measure-Object -Property n_genome_metabolome_links -Sum).Sum
                'Reported BGC-MS2 records: ' + ($projects | Measure-Object -Property n_bgc_ms2_links -Sum).Sum
                break
            }
            '1b' {
                Import-Csv $links |
                    Where-Object { [double]$_.actino_fraction -ge 0.5 } |
                    Select-Object -First 8 metabolomics_id, mibig, known_link |
                    Format-Table -Wrap -AutoSize
                break
            }
            '2' {
                Get-Content $tutorial
                break
            }
            '2b' {
                Select-String -Path $results -Pattern '^\| (spectra with peaks|spectra \(protocol\)|share \(protocol\)|spectra \(all_kept\)) \|' |
                    Select-Object -First 4 -ExpandProperty Line
                break
            }
            '3' {
                Select-String -Path $results -Pattern '^\| A1 family recovery@5:' |
                    Select-Object -First 5 -ExpandProperty Line
                break
            }
            '3b' {
                Select-String -Path $results -Pattern '^\| (B independent compounds after all checks|B1 Spearman with Tanimoto:)' |
                    Select-Object -First 5 -ExpandProperty Line
                break
            }
            '4' {
                Select-String -Path $federated -Pattern '^\| labs \| (FedAvg|centralised)' |
                    Select-Object -First 2 -ExpandProperty Line
                break
            }
            '5' {
                Select-String -Path $feasibility -Pattern '^\| MSV000078836 \|' |
                    Select-Object -First 1 -ExpandProperty Line
                break
            }
            '5b' {
                Start-Process 'https://github.com/Victoria125/priority-wur/pull/1'
                break
            }
            default {
                Write-Host 'Available demos: 1, 1b, 2, 2b, 3, 3b, 4, 5, 5b'
            }
        }
    } catch {
        Write-Host ('Could not show demo ' + $n + ': ' + $_.Exception.Message)
        Write-Host 'Check that demo.ps1 is inside priority-wur\dreams_molecular_families.'
    }
}

if ($Part) {
    demo $Part
}
