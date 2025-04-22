import math
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # filepath = "C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\P40\\p40_snr_static.txt"
    # filepath = "C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\m11pro\\m11pro_snr_static.txt"
    filepath = "C:\\Users\\zhang\\Desktop\\paper\\data_analysis\\oneplusresult\\oneplus12_snr_static.txt"

    f = open(filepath, 'r')  # Add the derived file path

    ID = []
    AZ = []
    EL = []

    ln = f.readline()
    while ln:
        ln = f.readline()
        if not ln:
            break
        str = ln.split()

        id = str[2]
        az = float(str[3]) / 180 * math.pi
        el = 90 - float(str[4])

        ID.append(id)
        AZ.append(az)
        EL.append(el)

    f.close()
    ax = plt.subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])

    satnum = 32
    if (ID[0][0] == 'G'):
        satnum = 32
    elif (ID[0][0] == 'R'):
        satnum = 27
    elif (ID[0][0] == 'E'):
        satnum = 36
    elif (ID[0][0] == 'C'):
        satnum = 54
    elif(ID[0][0] == 'J'):
        satnum = 2

    SATAZ = []
    SATEL = []
    SATID = []
    for i in range(1, satnum):
        for j in range(0, len(ID)):
            if int(ID[j][1:3]) == i:
                az = AZ[j]
                SATAZ.append(az)
                el = EL[j]
                SATEL.append(el)
                id = ID[j]
                SATID.append(id)
        if len(SATAZ) > 0:
            c = ax.scatter(SATAZ, SATEL, s=1, marker=".", alpha=0.75)

            ax.text(SATAZ[0], SATEL[0], SATID[0])
            SATAZ = []
            SATEL = []
            SATID = []

    ax.yaxis.set_label_position('right')
    ax.tick_params('y', labelleft=False)
    plt.savefig('skyplot.jpg', dpi=300)
    plt.show()