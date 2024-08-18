import numpy as np
import cv2

# Funkcja do tworzenia kernela na podstawie rozmiaru i rodzaju
def create_kernel(size, rodzaj):
    """
    Funkcja generująca kernel (macierz filtru) o określonym rozmiarze i rodzaju.

    Parameters:
    - size: Rozmiar kernela (musi być liczbą nieparzystą).
    - rodzaj: Rodzaj kernela ("Gaussian" dla filtru Gaussa lub "Box" dla filtru średniej).

    Returns:
    Numpy array reprezentujący kernel.
    """
    if size % 2 == 0:
        raise TypeError("Rozmiar kernela musi być liczbą nieparzystą")
    else:
        if rodzaj == "Gaussian":
            if size == 3:
                return np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])
            elif size == 5:
                return np.array([[1, 4,  6 , 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]])
        elif rodzaj == "Box":
            return np.ones((size, size))

# Funkcja do filtracji medianowej
def median_filtr(szum, kernel_size):
    """
    Filtr medianowy, który redukuje szum poprzez zastępowanie wartości pikseli medianą z określonego otoczenia.

    Parameters:
    - szum: Obraz wejściowy zawierający szum.
    - kernel_size: Rozmiar okna do obliczania mediany (musi być liczbą nieparzystą).

    Returns:
    Obraz po zastosowaniu filtracji medianowej.
    """
    height, width, dim = szum.shape
    # Tworzenie obramowania wokół obrazu
    padded = cv2.copyMakeBorder(szum, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)

    # Inicjalizacja obrazu wynikowego
    filtered = np.zeros((height, width, dim), dtype=np.uint8)

    # Iteracja przez każdy piksel obrazu
    for x in range(height):
        for y in range(width):
            # Wybór obszaru o rozmiarze kernela
            area = padded[x:x+kernel_size, y:y+kernel_size, :]

            # Zastosowanie mediany do każdego kanału koloru
            filtered[x, y, 0] = np.median(area[:,:,0])
            filtered[x, y, 1] = np.median(area[:,:,1])
            filtered[x, y, 2] = np.median(area[:,:,2])
    return filtered

# Funkcja do obliczania PSNR (Peak Signal-to-Noise Ratio)
def psnr_db(img1, img2):
    """
    Oblicza wartość PSNR (Peak Signal-to-Noise Ratio) między dwoma obrazami.

    Parameters:
    - img1: Pierwszy obraz.
    - img2: Drugi obraz.

    Returns:
    Wartość PSNR w skali decybelowej.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return round(10 * np.log10(PIXEL_MAX**2 / mse), 5)

# Wczytanie obrazów
working_image = cv2.imread("Leopard_with_noise.jpg")
no_noise_image = cv2.imread("Leopard.jpg")

# Filtracja medianowa z różnymi rozmiarami kernela
median_border_replicate_10 = median_filtr(working_image, 10)
cv2.imwrite(r"Wyniki\roznica_median_border_replicate_10.png", (no_noise_image - median_border_replicate_10) ** 2)
cv2.imshow("Różnica Median Border Replicate 10", cv2.convertScaleAbs(no_noise_image - median_border_replicate_10))

median_border_replicate_3 = median_filtr(working_image, 3)
median_border_replicate_5 = median_filtr(working_image, 5)
median_border_replicate_7 = median_filtr(working_image, 7)

# Wizualizacja różnic dla różnych rozmiarów kernela
cv2.imwrite(r"Wyniki\roznica_median_border_replicate_3.png", (no_noise_image - median_border_replicate_3) ** 2)
cv2.imshow("Różnica Median Border Replicate 3", cv2.convertScaleAbs(no_noise_image - median_border_replicate_3))

cv2.imwrite(r"Wyniki\roznica_median_border_replicate_5.png", (no_noise_image - median_border_replicate_5) ** 2)
cv2.imshow("Różnica Median Border Replicate 5", cv2.convertScaleAbs(no_noise_image - median_border_replicate_5))

cv2.imwrite(r"Wyniki\roznica_median_border_replicate_7.png", (no_noise_image - median_border_replicate_7) ** 2)
cv2.imshow("Różnica Median Border Replicate 7", cv2.convertScaleAbs(no_noise_image - median_border_replicate_7))

# Filtracja medianowa o rozmiarze 3x3, 5x5 i 7x7
print("Rozmiar jądra: 3x3")
print("border_replicate: ", psnr_db(no_noise_image, median_border_replicate_3), "dB", "vs cv2", psnr_db(median_border_replicate_3, cv2.medianBlur(working_image, 3)))
cv2.imwrite(r"Wyniki\median_border_replicate_3.png", median_border_replicate_3)
cv2.imshow("Median Border Replicate 3", median_border_replicate_3)

print("Rozmiar jądra: 5x5")
print("border_replicate: ", psnr_db(no_noise_image, median_border_replicate_5), "dB", "vs cv2", psnr_db(median_border_replicate_5, cv2.medianBlur(working_image, 5)))
cv2.imwrite(r"Wyniki\median_border_replicate_5.png", median_border_replicate_5)
cv2.imshow("Median Border Replicate 5", median_border_replicate_5)

print("Rozmiar jądra: 7x7")
print("border_replicate: ", psnr_db(no_noise_image, median_border_replicate_7), "dB", "vs cv2", psnr_db(median_border_replicate_7, cv2.medianBlur(working_image, 7)))
cv2.imwrite("Wyniki\median_border_replicate_7.png", median_border_replicate_7)
cv2.imshow("Median Border Replicate 7", median_border_replicate_7)

# Czekaj na naciśnięcie klawisza i zamknij wszystkie okna
cv2.waitKey(0)
cv2.destroyAllWindows()
