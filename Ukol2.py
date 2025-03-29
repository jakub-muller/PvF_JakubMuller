import numpy as np
import matplotlib.pyplot as plt

def histogramovator(data): #spocte histogram
    mez1=np.min(data)
    mez2=np.max(data)
    H=100
    suma = 0
    histogram = np.zeros(H)
    for i in range(len(data)):
        index = int((data[i] - mez1) / (mez2 - mez1) * H)
        if index == H:
            index = H - 1
        histogram[index] += 1
        suma += 1
    histogram = histogram / suma
    return histogram

def kreslic_dvou_rozdeleni(data1, data2): #nakresli dva histogramy vedle sebe
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for data1
    mez1 = np.min(data1)
    mez2 = np.max(data1)
    H = 100
    bin_edges = np.linspace(mez1, mez2, len(data1) + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[0].bar(bin_centers, data1, width=(mez2 - mez1) / len(data1), color='skyblue', edgecolor='black', alpha=0.7)
    axs[0].set_title("Histogram 1", fontsize=16, fontweight='bold')
    axs[0].set_xlabel("Data values", fontsize=14)
    axs[0].set_ylabel("Frequency", fontsize=14)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot for data2
    mez1 = np.min(data2)
    mez2 = np.max(data2)
    H = int(np.log2(len(data2)) + 1)
    bin_edges = np.linspace(mez1, mez2, len(data2) + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[1].bar(bin_centers, data2, width=(mez2 - mez1) / len(data2), color='salmon', edgecolor='black', alpha=0.7)
    axs[1].set_title("Histogram 2", fontsize=16, fontweight='bold')
    axs[1].set_xlabel("Data values", fontsize=14)
    axs[1].set_ylabel("Frequency", fontsize=14)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def kreslic(data): #nakresli histogram
    mez1=np.min(data)
    mez2=np.max(data)
    H=int(np.log2(len(data))+1)
    bin_edges = np.linspace(mez1, mez2, len(data) + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, data, width=(mez2 - mez1) / len(data), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Fancy Histogram", fontsize=16, fontweight='bold')
    plt.xlabel("Data values", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def kontrolni_suma(data): #spocte sumu pro kontrolu normovaci podminky
    suma = 0
    for i in range(len(data)):
        suma += data[i]
    print(f'{suma:+9.6f}')

def spocti_rozptyl(data): #spocte rozptyl
    suma = 0
    for i in range(len(data)):
        suma += data[i]
    suma = suma / len(data)
    rozptyl = 0
    for i in range(len(data)):
        rozptyl += (data[i] - suma) ** 2
    rozptyl = rozptyl / len(data)
    print(rozptyl)

def spocti_stredni_hodnotu(data): #spocte stredni hodnotu
    suma = 0
    for i in range(len(data)):
        suma += data[i]
    suma = suma / len(data)
    print(suma)

def komplet_rozbor_vyberu(data): #spocte histogra, nakresli, spocte sumu pro normovaci podminku, spocte rozptyl a stredni hodnotu
    kontrolni_suma(histogramovator(data))
    spocti_rozptyl(data)
    spocti_stredni_hodnotu(data)

def rovnomerne_rozdeleni(n): #n je pocet nahodnych cisel
    rozdeleni = np.random.rand(n)
    return rozdeleni

def logisticka_mapa(x0,n,r=4):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x

def gausovske_rozdeleni(mi=6,sigma=1,n=10000): #mi je stredni hodnota, sigma je rozptyl, n je pocet nahodnych cisel
    rozdeleni = np.random.normal(mi, sigma, n)
    return rozdeleni

def soucet_rozdeleni(rozdeleni, n, M=12): #rozdeleni je funkce, ktera generuje nahodna cisla, n je pocet nahodnych cisel, N je pocet scitani
    soucet = np.zeros(n)
    for i in range(M):
        rozdeleni_n=rovnomerne_rozdeleni(n)
        soucet+=rozdeleni_n
    return soucet

#UKOL 1

komplet_rozbor_vyberu(rovnomerne_rozdeleni(10000))
kreslic(histogramovator(rovnomerne_rozdeleni(10000)))

#UKOL 2

komplet_rozbor_vyberu(soucet_rozdeleni(rovnomerne_rozdeleni,10000,2))
kreslic(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,2)))

#UKOL 3

kreslic_dvou_rozdeleni(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,12)),histogramovator(gausovske_rozdeleni(6,1,10000)))
komplet_rozbor_vyberu(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,12)))
komplet_rozbor_vyberu(histogramovator(gausovske_rozdeleni(6,1,10000)))


# Tedy pro hodnotu M=12 jsou oba histogramy zaměnitelné, pro odlišnou hodnotu M se liší, lze taky ověřit:

#kreslic_dvou_rozdeleni(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,5)),histogramovator(gausovske_rozdeleni(6,1,10000)))
#komplet_rozbor_vyberu(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,5)))
#komplet_rozbor_vyberu(histogramovator(gausovske_rozdeleni(6,1,10000)))

#kreslic_dvou_rozdeleni(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,30)),histogramovator(gausovske_rozdeleni(6,1,10000)))
#komplet_rozbor_vyberu(histogramovator(soucet_rozdeleni(rovnomerne_rozdeleni,10000,30)))
#komplet_rozbor_vyberu(histogramovator(gausovske_rozdeleni(6,1,10000)))

#Zároveň lze součet 12ti rovnoměrných rozdělení považovat za generátor náhodných čísel s gausovnským rozdělením, což je odpověď na otázku z úkolu 3. 

#UKOL 4

kreslic(histogramovator(logisticka_mapa(0.6,10000,4)))
#tedy ne, logistické zobrazení není dobrým generátorem náhodných čísel s rovnoměrným rozdělením, protože histogram není rovnoměrný, hodnoty na okraji jsou častější než hodnoty uprostřed
#a zároveň nesahá po celém <0,1> intervalu, což je součást otázky

#UKOL 5

import numpy as np

def generate_random_matrix(n=10000, sigma=1.0):
    
    A = np.random.normal(loc=0, scale=sigma, size=(n, n))
    return A


def spocti_eigenvalues(n, sigma):
    matrix = generate_random_matrix(n, sigma)
    eigenvalues = np.linalg.eigvals(matrix)
    return eigenvalues

def transponuj_matici(matrix):
    return np.transpose(matrix)

Matice=0.5*(generate_random_matrix(1000,1)+transponuj_matici(generate_random_matrix(1000,1)))
eigenvalues = np.linalg.eigvals(Matice)

import numpy as np

def sample_semicircle(num_samples, sigma, N):
    R = 2 * sigma * np.sqrt(N)
    # Maximum value of the density occurs at a = 0:
    max_pdf = 2 / (np.pi * R)
    
    samples = []
    
    # Rejection sampling loop
    while len(samples) < num_samples:
        # Generate a batch of candidate points
        batch_size = num_samples - len(samples)
        x_candidates = np.random.uniform(-R, R, size=batch_size)
        # Uniform random numbers for rejection test, in the range [0, max_pdf]
        u = np.random.uniform(0, max_pdf, size=batch_size)
        # Evaluate the Wigner semicircle pdf for the candidates
        pdf_vals = (2 / (np.pi * R**2)) * np.sqrt(R**2 - x_candidates**2)
        # Accept candidates where u < pdf(x)
        accepted = x_candidates[u < pdf_vals]
        samples.extend(accepted.tolist())
    
    # Return exactly num_samples values (in case we oversampled)
    return np.array(samples[:num_samples])

sigma = 1.0
N = 1000
num_samples = 10000
    
samples = sample_semicircle(num_samples, sigma, N)

kreslic_dvou_rozdeleni(histogramovator(eigenvalues),histogramovator(samples))
komplet_rozbor_vyberu(histogramovator(eigenvalues))
komplet_rozbor_vyberu(histogramovator(samples))

#dle grafu, střední hodnotě a rozptylu lze zhruba říci, že jsou si rozdělení podobná
#je ale třeba si uvědomit, že mnoho vlastních čísel program změní, protože mají imaginární část. Ta je anulována, aby se vešly do histogramu
