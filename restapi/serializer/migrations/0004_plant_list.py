# Generated by Django 2.1.5 on 2019-01-20 18:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('serializer', '0003_users'),
    ]

    operations = [
        migrations.CreateModel(
            name='Plant_List',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('plant_name', models.CharField(max_length=20)),
                ('specific_plant', models.CharField(max_length=20)),
                ('plant_width', models.IntegerField()),
                ('plot_size', models.IntegerField()),
                ('plant_distance', models.IntegerField()),
            ],
        ),
    ]