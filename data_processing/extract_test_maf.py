import argparse

def extract_first_three_blocks(maf_file, output_file):
    alignment_blocks = []
    current_block = []
    block_count = 0

    with open(maf_file, 'r') as f:
        for line in f:
            if line.startswith('a'):
                if current_block:
                    alignment_blocks.append(current_block)
                    block_count += 1
                    if block_count == 3:
                        break
                current_block = [line]
            elif line.startswith('s') or line.strip() == '':
                current_block.append(line)
        
        # Add the last block if it was not added
        if current_block and block_count < 3:
            alignment_blocks.append(current_block)

    with open(output_file, 'w') as f:
        for block in alignment_blocks:
            for line in block:
                f.write(line)

def main():
    parser = argparse.ArgumentParser(
        description="Extract the first three alignment blocks from a MAF file"
    )
    parser.add_argument(
        "--maf_file",
        type=str,
        default="/data/mica/proj/uppstore2017228/KLT.04.200M/200m_MD/data/new_250_MAMMALS_v2_20201120/MAF/HUMAN/241MAMMALS/human_onlychr_v2_mdong_10Mb_241MAMMALS/chr1.150000000.10000000.maf",
        help="Path to the MAF file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='/home/t-mconsens/gamba/data_processing/data/240-mammalian/test_maf.maf',
        help="Path to the output MAF file",
    )
    args = parser.parse_args()

    extract_first_three_blocks(args.maf_file, args.output_file)
    print(f"Extracted the first three alignment blocks from {args.maf_file} and saved to {args.output_file}")

if __name__ == "__main__":
    main()