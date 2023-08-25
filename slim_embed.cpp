#include <stdio.h>
#include <vector>

int main() {
    // read binary file
    FILE* src_f = fopen("onnx/embed.weight", "rb");
    constexpr size_t num = 4096 * 151936;
    // constexpr size_t slim_num = 4096 * 123791;
    constexpr size_t slim_num = num;
    std::vector<float> src_buffer(num);

    fread(src_buffer.data(), 1, slim_num * sizeof(float), src_f);
    fclose(src_f);

    // convert to bf16
    std::vector<short> dst_buffer(slim_num);
    for (int i = 0; i < slim_num; i++) {
        dst_buffer[i] = reinterpret_cast<short*>(src_buffer.data())[2 * i + 1];
        if (i < 10) {
            float tmp;
            reinterpret_cast<short*>(&tmp)[0] = 0;
            reinterpret_cast<short*>(&tmp)[1] = dst_buffer[i];
            printf("%f -> %f\n", src_buffer[i], tmp);
        }
    }
    // write to bianry file
    FILE* dst_f = fopen("slim_word_embeddings_bf16.bin", "wb");
    fwrite(dst_buffer.data(), 1, slim_num * sizeof(short), dst_f);
    fclose(dst_f);
    return 0;
}