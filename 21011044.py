# numpy'ı dizi işlemleri ve sayısal hesaplamalar yapmak için
import numpy as np
# matplotlib kütüphanesini grafik çizimleri ve görselleştirme yapmak için
import matplotlib.pyplot as plt
# PIL kütüphanesinden görüntü işlemleri gerçekleştirmek için kullandım
from PIL import Image

POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1

# load_images fonksiyonunu verilen dosya yollarından resimleri yükleyip işleyerek binary hale getirmek için kullandım
def load_images(image_paths):
    # İşlenmiş resimleri saklamak amacıyla boş bir liste oluşturdum
    images = []
    # Her dosya yolunu işlemek için döngü
    for path in image_paths:
        # Belirtilen dosya yolundan resmi açıp gri tona çevirme işlemi
        img = Image.open(path).convert("L")
        # Resmi 24x24 piksel boyutuna indirgeme
        img = img.resize((24, 24))
        # PIL nesnesini numpy dizisine dönüştürme işlemi
        img_arr = np.array(img)
        # Resimdeki piksel değerlerini 128 eşiğine göre binary yapıyoruz
        binary_img = (img_arr >= 128).astype(int)
        # İşlenmiş binary resmi listeye ekle
        images.append(binary_img)
    return images

# initialize_population fonksiyonunu popülasyonda her biri 7 adet 3x3 binary pattern oluşturmak için kullanıcaz
def initialize_population(pop_size):
    # Her biri 7 adet 3x3 matris içeren bireyleri rastgele oluşturuyorum
    population = [np.random.randint(0, 2, (7, 3, 3)) for _ in range(pop_size)]
    return population

# dive_blocks fonksiyonunu 24x24 boyutundaki bir görüntüyü 3x3 bloklara bölmek için kullanılacaktır
def dive_blocks(image, block_size=3):
    # Resmi 8 blok olacak şekilde yeniden boyutlandırıp transpose ve reshape işlemleri ile 3x3 bloklara ayırmamız gerekiyor
    blocks = image.reshape(8, block_size, 8, block_size) \
                 .transpose(0, 2, 1, 3) \
                 .reshape(-1, block_size, block_size)
    return blocks

# evaluate_individual fonksiyonunu bir bireyin loss değeri üzerinden değerlendirmek için kullanacakım
def evaluate_individual(individual, images):
    # Toplam loss değerini 0
    total_loss = 0
    # Her bir görüntü üzerinde döngü
    for img in images:
        # Görüntüyü 3x3 bloklara bölmemiz lazım
        blocks = dive_blocks(img)
        # Her bir blok için döngü
        for block in blocks:
            # Bireyin her bir patterni ile bloğun farkını hesaplayıp bir liste oluşturma işlemi
            differences = [np.sum(np.abs(pattern - block)) for pattern in individual]
            # O blok için en düşük farkıtoplam lossa ekleyerek en iyisini buluyoruz
            total_loss += min(differences)
    return total_loss

# selection fonksiyonunu roulette wheel yöntemiyle fitness değerine göre seçim yapmamız lazım
def selection(population, fitnesses):
    # Fitness değerlerini numpy dizisi haline getirerek işe başlarız
    fitnesses = np.array(fitnesses)
    # Sıfıra bölme hatasını önlemek amacıyla küçük bir eps değeri eklememiz gerekiyor kısır döngüye girmemesi için
    eps = 1e-6
    # Düşük loss yani yüksek fitness olan bireylerin seçilme olasılığını artırmak için ters loss
    probabilities = 1 / (fitnesses + eps)
    # Olasılıkların toplamlarının 1 olması gerekli
    probabilities = probabilities / probabilities.sum()
    # Bu olasılıklara göre indeksleri rastgele seçeriz roulette wheell ile
    indices = np.random.choice(len(population), size=len(population), p=probabilities)
    # Seçilen indekslere göre yeni popülasyonu oluşturarak döndürüyorum
    return [population[i] for i in indices]

# crossover fonksiyonunu iki ebeveyn arasında tek noktalı crossover işlemi yaparak iki çocuk birey oluşturmaya yarayacak
def crossover(parent1, parent2):
    # 1 ile 6 arasında rastgele bir nokta seçreiz
    point = np.random.randint(1, 7)
    # Ebeveyn1 ilk nokta kadar patterni ile ebeveyn2nin kalan patternlerını birleştirerek ilk çocuğu oluşturdum
    child1 = np.concatenate((parent1[:point], parent2[point:]), axis=0)
    # Ebeveyn2nin ilk nokta kadar patterni ile ebeveyn1in kalan patternlerini birleştirerek ikinci çocuğu oluşturdum
    child2 = np.concatenate((parent2[:point], parent1[point:]), axis=0)
    # Oluşturulan iki çocuğu döndürürüz
    return child1, child2

# mutate fonksiyonunu bireydeki her bit için mutasyon oranına göre rastgele mutasyon uygulamamız gerekiyor
def mutate(individual, mutation_rate):
    # Bireyin tüm patternleri
    for i in range(individual.shape[0]):
        # Her patternin satırları üzerinde dönmeli
        for j in range(individual.shape[1]):
            # Her satırın sütunları üzerinde dönmeli
            for k in range(individual.shape[2]):
                # Rastgele oluşturulan değerin mutasyon oranından küçük olup olmadığını kontrol etmemiz gerekiyor
                if np.random.rand() < mutation_rate:
                    # 0 olan biti 1 ve 1 olan biti 0 yaparak mutasyona uğratırız
                    individual[i, j, k] = 1 - individual[i, j, k]
    return individual

# genetic_algorithm fonksiyonunu genetik algoritmanın tüm adımlarını uygulayarak bize en iyi bireyleri verir
def genetic_algorithm(images, pop_size, num_generations, mutation_rate, record_interval=10):
    # Başlangıç popülasyonunu rastgele oluşturuyoruz
    population = initialize_population(pop_size)
    # Her jenerasyonda elde edilen en iyi loss değerlerini bir listede saklamamız gerekiyor
    best_losses = []
    # Belirli jenerasyonlardaki en iyi bireyleri saklamamız gerkiyor
    best_history = {}
    # En iyi birey ve en düşük loss değerini başlangıçta bir önemi olmayacak sonradan güncellenecek
    best_individual = None
    best_loss = np.inf

    # Jenerasyon sayısı kadar döngü dönmelidir
    for gen in range(num_generations):
        # Her bireyin loss değerini hesaplamak için evaluate_individual i kullanırız
        fitnesses = [evaluate_individual(ind, images) for ind in population]
        # O jenerasyonda elde edilen en düşük loss değeri
        gen_best_loss = min(fitnesses)
        # Bu jenerasyonun en iyi loss değeri listeye ekle
        best_losses.append(gen_best_loss)
        # Mevcut jenerasyondaki en iyi loss değeri şimdiye kadarki en iyi losstan düşükse güncelleme yapıyrum
        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_individual = population[np.argmin(fitnesses)]
        # Belirlenen aralıkta veya son jenerasyonda en iyi bireyi kaydederek best_historye ekledim
        if (gen % record_interval == 0) or (gen == num_generations - 1):
            best_history[gen] = population[np.argmin(fitnesses)]
        # Jenerasyonun en iyi loss değerini ekrana yazdırırız
        print(f"Generation {gen + 1}: Best Loss = {gen_best_loss}")

        # Roulette wheel yöntemiyle yeni ebeveyn popülasyonu seçimi
        selected = selection(population, fitnesses)
        # Yeni nesil popülasyonu oluşturmak için boş bir liste lazım
        next_population = []
        # Popülasyonu çiftler halinde işlemek için ikişer artışla döngüye girmelidir
        for i in range(0, pop_size, 2):
            # İlk ebeveyn
            parent1 = selected[i]
            # İkinci ebeveyni
            parent2 = selected[(i + 1) % pop_size] #dizi sınırını aşmamalıdır
            # Ebeveynler arasında crossover işlemi uygulama ve çocuk oluşumu iki tane
            child1, child2 = crossover(parent1, parent2)
            # İlk çocuğa mutasyon
            child1 = mutate(child1, mutation_rate)
            # İkinci çocuğa mutasyon
            child2 = mutate(child2, mutation_rate)
            # Oluşturulan iki çocuğu yeni popülasyona eklee
            next_population.extend([child1, child2])
        # Yeni oluşturulan popülasyonu mevcut popülasyon olarak ayarla
        population = next_population

    # En iyi birey jenerasyonlardaki loss evrimi ve best historyyi döndürüyorum
    return best_individual, best_losses, best_history

# plot_loss fonksiyonunu jenerasyonlardaki en iyi loss değerlerini grafik üzerinde gösterme işlemi
def plot_loss(best_losses):
    plt.figure()
    plt.plot(best_losses, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.title("Genetic Algorithm: Best Loss per Generation")
    plt.grid(True)
    plt.show()

# plot_similarity fonksiyonunu loss değerlerini benzerlik oranına çevirip grafik üzerinde göstermemizi sağlıyor
def plot_similarity(best_losses, num_images=5):
    # Maksimum loss değerini hesapladım
    max_loss = 9 * 64 * num_images
    # Her jenerasyon için benzerlik oranını hesaplayarak listeye dönüştürdüm
    similarities = [1 - (loss / max_loss) for loss in best_losses]
    plt.figure()
    # Benzerlik oranlarını yeşil renkte ve nokta işareti ile göster
    plt.plot(similarities, marker='o', color='green')
    plt.xlabel("Generation")
    plt.ylabel("Similarity Ratio")
    plt.title("Evolution of Similarity Ratio Across Generations")
    plt.grid(True)
    plt.show()

# plot_final_patterns fonksiyonunu en iyi bireye ait 7 pattern i görselleştirilmesini sağlar
def plot_final_patterns(best_individual):
    fig, axes = plt.subplots(1, 7, figsize=(15, 3))
    for i in range(7):
        axes[i].imshow(best_individual[i], cmap='gray')
        axes[i].set_title(f"Pattern {i + 1}")
        axes[i].axis("off")
    plt.show()

# plot_pattern_evolution fonksiyonunu belirli jenerasyonlardaki en iyi bireylerin pattern evrimini gösterir
def plot_pattern_evolution(best_history):
    generations = sorted(best_history.keys())
    num_records = len(generations)
    fig, axes = plt.subplots(num_records, 7, figsize=(15, 2*num_records))
    for row, gen in enumerate(generations):
        individual = best_history[gen]
        for col in range(7):
            if num_records > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            ax.imshow(individual[col], cmap='gray')
            if col == 0:
                ax.set_ylabel(f"Gen {gen}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle("Evolution of Best Patterns Across Generations")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# experiment_mutation_rates fonksiyonunu farklı mutasyon oranları için genetik algoritmayı çalıştırıp sonuçları karşılaştırmamız için oluşturdum
def experiment_mutation_rates(images, pop_size, num_generations, mutation_rates):
    # Her mutasyon oranı için loss evrimlerini saklamak amacıyla boş bir sözlük gerekli
    results = {}
    # Denenecek her mutasyon oranı için döngü
    for m in mutation_rates:
        # Hangi mutasyon oranı ile deney yapıldığını ekrana yazdır
        print(f"Running experiment for mutation rate {m}")
        # Genetik algoritmayı çalıştırarak loss evrimini hesapla kaydet
        _, losses, _ = genetic_algorithm(images, pop_size, num_generations, m, record_interval=num_generations)
        # Elde edilen loss evrimlerini sonuç sözlüğüne ekle
        results[m] = losses
    plt.figure()
    for m, losses in results.items():
        plt.plot(losses, marker='o', label=f"Mutation Rate {m}")
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.title("Effect of Mutation Rate on Best Loss Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()

# reconstruct_image fonksiyonunu orijinal 24x24 görüntüyü en iyi bireyin patternlerini kullanarak yeniden oluşturmak için kullanıyoruz
def reconstruct_image(image, best_individual):
    # Görüntüyü 3x3 bloklara bölmek için dive_blocks fonksiyonunu kullanılır
    blocks = dive_blocks(image)
    # Yeniden oluşturulacak blokları saklamak amacıyla boş bir liste oluşturdum
    new_blocks = []
    # Her bir bloğu işlemek için döngü
    for block in blocks:
        # Her pattern ile blok arasındaki farkı hesaplayıp bir liste oluştur
        differences = [np.sum(np.abs(pattern - block)) for pattern in best_individual]
        # En iyi uyum sağlayan patternin indeksini belirle al
        best_index = np.argmin(differences)
        # Seçilen patterni yeni bloklar listesine ekleme işlemini yap
        new_blocks.append(best_individual[best_index])
    # 64 adet 3x3 bloğu 8x8lik yapıda numpy dizisine dönüştür
    new_blocks = np.array(new_blocks).reshape(8, 8, 3, 3)
    # Blokları transpose edip yeniden boyutlandırarak 24x24lük yeni resmi oluşturdum
    new_blocks = new_blocks.transpose(0, 2, 1, 3).reshape(24, 24)
    # Yeniden oluşturulan resmi döndürüyorum
    return new_blocks

# Ana program akışı
if __name__ == '__main__':
    image_files = [
        r"binary_image_1.png",
        r"binary_image_2.png",
        r"binary_image_3.png",
        r"binary_image_4.png",
        r"binary_image_5.png"
    ]
    # Belirlenen dosya yollarından resimleri yükleyip binary hale getirme
    images = load_images(image_files)
    # Resimlerin yüklendiğini ve genetik algoritmanın çalıştırılacağını ekrana yazdırmamız lazım
    print("Görseller yüklendi. Genetik Algoritma çalıştırılıyor...")
    # Genetik algoritmayı çalıştırarak en iyi birey loss evrimi ve best historyyi elde ediyoruz
    best_individual, best_losses, best_history = genetic_algorithm(
        images, POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, record_interval=10
    )
    # Jenerasyonlardaki en iyi loss değerlerini grafik üzerinde görselleştir
    plot_loss(best_losses)
    # Resim sayısına bağlı olarak benzerlik oranlarının evrimini grafik üzerinde göster
    plot_similarity(best_losses, num_images=len(images))
    # En iyi bireye ait 7 patterni grafik üzerinde görselleştirerek
    plot_final_patterns(best_individual)
    # Belirli jenerasyonlardaki en iyi bireylerin pattern evrimini grafik üzerinde gösterdim
    plot_pattern_evolution(best_history)
    # En iyi patternleri dosya olarak kaydetmemiz lazım
    for i in range(7):
        # En iyi bireyin patternini 0-255 ölçeğine çevirip görüntü nesnesine dönüştürmemiz lazım
        pattern_img = Image.fromarray((best_individual[i] * 255).astype(np.uint8))
        # Patterni "best_pattern_i.png" formatında dosya olarak kaydediyorum
        pattern_img.save(f"best_pattern_{i + 1}.png")
    # Patternlerin dosyaya kaydedildiğini ekrana yazdırdım
    print("En iyi pattern'ler 'best_pattern_1.png' ... 'best_pattern_7.png' olarak kaydedildi.")
    # Her bir orijinal resim için yeniden oluşturma işlemi yapmak üzere döngüye giriyoruz
    for idx, img in enumerate(images):
        # reconstruct_image fonksiyonunu kullanarak resmi en iyi bireyin patternleriyle yeniden oluşturma işlemi
        reconstructed = reconstruct_image(img, best_individual)
        # Yeniden oluşturulan resmi gösterme işlemi:
        plt.figure()
        # Resmi gri tonlarda grafik üzerine yerleştiririz
        plt.imshow(reconstructed, cmap='gray')
        # Grafiğe "Reconstructed İmage" başlığını ekledim
        plt.title(f"Reconstructed Image {idx + 1}")
        # Eksen çizgilerini kapat
        plt.axis("off")
        # Grafiği ekranda gösteriyorum
        plt.show()
        # Yeniden oluşturulan resmi 0-1 arası değerleri 0-255 ölçeğine çevirip görüntü nesnesine dönüştürmemzi lazım
        recon_img = Image.fromarray((reconstructed * 255).astype(np.uint8))
        # Resmi "reconstructed_image_i.png" formatında dosya olarak kaydettim
        recon_img.save(f"reconstructed_image_{idx + 1}.png")
    # Orijinal resimlerin, en iyi patternlerle yeniden oluşturulup dosyaya kaydedildiğini ekrana yazdırdım
    print("Orijinal resimler, en iyi patternlerle yeniden oluşturularak dosya olarak kaydedildi.")
    # Farklı mutasyon oranlarını [0.05, 0.1, 0.2] olarak belirledim
    mutation_rates = [0.05, 0.1, 0.2]
    # Farklı mutasyon oranlarıyla deney yaparak hiperparametre analizini gerçekleştiriyoruz
    experiment_mutation_rates(images, POPULATION_SIZE, NUM_GENERATIONS, mutation_rates)
